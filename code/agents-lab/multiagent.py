"""Part 7: one agent, or a lead agent that hands subtasks to worker agents.

  run_single   the Part 1 loop: one model, the data tools, one growing context window
  run_team     a lead agent whose only tool is `delegate`. Each delegate call starts a worker
               agent with a fresh context window, the data tools and nothing but the text the lead
               wrote; the worker's final reply comes back to the lead as the tool result. Delegate
               calls made in the same turn run in parallel.

Every API call goes through ledger.create. A shared semaphore caps the requests in flight.
harness.py, react.py and ledger.py are imported, not changed.
"""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Step, Tool, Transcript, run_tool  # noqa: E402

ADAPTIVE = {"type": "adaptive"}


class Limited:
    """Wraps an SDK client: at most `n` requests in flight across all threads, and per-thread timing."""
    def __init__(self, cl, n=3):
        self.cl, self.sem, self.local = cl, threading.Semaphore(n), threading.local()
        self.messages = self

    def count_tokens(self, **kw):
        with self.sem:
            return self.cl.messages.count_tokens(**kw)

    def create(self, **kw):
        with self.sem:
            t0 = time.perf_counter()
            resp = self.cl.messages.create(**kw)
            self.local.last = time.perf_counter() - t0
            return resp


@dataclass
class Run:
    role: str                 # "single", "lead" or "worker"
    answer: str = ""
    stop: str = "max_steps"
    steps: list = field(default_factory=list)
    workers: list = field(default_factory=list)   # Run objects of the workers a lead started
    brief: str = ""           # for a worker: the task text the lead wrote

    @property
    def cost(self):
        return sum(s.cost for s in self.steps) + sum(w.cost for w in self.workers)

    def all_steps(self):
        return self.steps + [s for w in self.workers for s in w.all_steps()]


def loop(task, tools, system, *, part, role, client, model="claude-haiku-5-5", effort="medium",
         max_steps=30, max_cost=0.10, max_tokens=8000, transcript=None, run=None):
    """The Part 1 loop, with the tool calls of one turn run concurrently."""
    tools = {t.name: t for t in tools}
    request = dict(model=model, max_tokens=max_tokens, tools=[t.schema() for t in tools.values()],
                   system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, thinking=ADAPTIVE,
                   cache_control={"type": "ephemeral"})
    log = Transcript(transcript)
    log.write(event="task", role=role, task=task, model=model, effort=effort, system=system)
    run = run or Run(role)
    messages = [{"role": "user", "content": task}]
    for n in range(1, max_steps + 1):
        if run.cost >= max_cost:
            run.stop = "max_cost"
            break
        resp = ledger.create(client, part, note=f"{part} {role} step {n}", messages=messages, **request)
        u, calls = resp.usage, [b for b in resp.content if b.type == "tool_use"]
        step = Step(n, u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                    u.output_tokens, round(client.local.last, 3), ledger.cost_of(model, u),
                    resp.stop_reason, [f"{b.name}({json.dumps(b.input)})" for b in calls])
        step.role, step.model = role, model
        run.steps.append(step)
        log.write(event="step", role=role, **{k: v for k, v in step.__dict__.items() if k != "role"},
                  content=[b.model_dump(exclude_none=True) for b in resp.content])
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            run.stop = resp.stop_reason
            run.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        with ThreadPoolExecutor(max(1, len(calls))) as ex:   # parallel delegate calls run at once
            results = list(ex.map(lambda b: run_tool(tools, b), calls))
        messages.append({"role": "user", "content": results})
        log.write(event="tool_results", role=role, n=n, results=results)
    log.write(event="result", role=role, answer=run.answer, stop=run.stop, steps=len(run.steps),
              cost=run.cost)
    return run


# ---- The two designs -----------------------------------------------------------------------

LEAD = """You are the lead analyst for Larkspur Robotics' incident tracker. You cannot read the tracker
yourself. Your only tool is `delegate`, which starts a worker agent that can search and read incident
records. A worker starts with an empty context: it knows nothing except the brief you write, and it
cannot see the user's question, your other briefs or other workers' reports.

How to work:
- Split the question into subtasks that do not depend on each other, and delegate them all in the same
  turn so they run in parallel. Use one worker for work that cannot be split (for example, a chain where
  each record names the next), and sequence subtasks only when one needs another's result.
- Write each brief so it stands alone: the exact services, months, incident IDs and fields involved,
  what to compute, and the exact format of the report you want back (for example, the value plus the
  incident IDs it counted).
- Check the reports against each other and the question before answering. If a report is missing
  something you need, delegate a follow-up.

When you are done, end your reply with one line in the form
ANSWER: <the answer only>"""

WORKER = """You are a worker analyst on Larkspur Robotics' incident tracker. A lead analyst gave you the
task below. Use the tools to read every record the task needs and never guess a value you have not read.
Your final reply goes back to the lead, who cannot see your tool calls or their results, so it must
contain everything the lead asked for, in the format it asked for. Keep it short."""


def make_delegate(worker_tools, *, part, client, lead_run, model="claude-haiku-5-5", effort="medium",
                  max_cost=0.06, transcript_dir=None):
    count = iter(range(1, 1000))
    lock = threading.Lock()

    def delegate(task):
        with lock:
            k = next(count)
            w = Run("worker", brief=task)
            lead_run.workers.append(w)
        path = transcript_dir / f"worker{k:02d}.jsonl" if transcript_dir else None
        loop(task, worker_tools, WORKER, part=part, role=f"worker{k}", client=client, model=model,
             effort=effort, max_steps=30, max_cost=max_cost, transcript=path, run=w)
        if w.stop != "end_turn":
            return f"[worker stopped early: {w.stop}] {w.answer}".strip()
        return w.answer

    return Tool("delegate", "Start a worker agent with a fresh context and the incident tracker tools. "
                "It sees only the brief you write here and returns its final report.",
                {"task": {"type": "string", "description": "The complete, self-contained brief for the worker"}},
                delegate)


def run_single(question, tools, system, *, part, client, model="claude-haiku-5-5", effort="medium",
               max_cost=0.25, transcript=None):
    return loop(question, tools, system, part=part, role="single", client=client, model=model,
                effort=effort, max_steps=40, max_cost=max_cost, transcript=transcript)


def run_team(question, worker_tools, *, part, client, lead_model="claude-haiku-5-5",
             worker_model="claude-haiku-5-5", effort="medium", max_cost=0.25, worker_max_cost=0.06,
             transcript_dir=None):
    if transcript_dir:
        Path(transcript_dir).mkdir(parents=True, exist_ok=True)
    lead = Run("lead")
    tool = make_delegate(worker_tools, part=part, client=client, lead_run=lead, model=worker_model,
                         effort=effort, max_cost=worker_max_cost,
                         transcript_dir=Path(transcript_dir) if transcript_dir else None)
    return loop(question, [tool], LEAD, part=part, role="lead", client=client, model=lead_model,
                effort=effort, max_steps=12, max_cost=max_cost,
                transcript=Path(transcript_dir) / "lead.jsonl" if transcript_dir else None, run=lead)
