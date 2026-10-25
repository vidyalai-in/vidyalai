"""A minimal agent loop on the Claude API: the shared lab harness for the Agentic AI series.

An agent here is a model in a loop: send the conversation and the tool list, run every tool
the model asks for, send all the results back in one user message, and repeat until the
model ends its turn or a step or cost limit stops it. Every API call goes through
ledger.create, which checks the budget and records the cost.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402


# ---- Tools: plain Python functions plus a JSON schema --------------------------------

@dataclass
class Tool:
    name: str
    description: str
    params: dict          # JSON Schema "properties"; every parameter is required
    fn: Callable[..., str]

    def schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,  # tool inputs are guaranteed to match the schema
            "input_schema": {"type": "object", "properties": self.params,
                             "required": list(self.params), "additionalProperties": False},
        }


def tool(description, **params):
    """Decorator: @tool("what it does", path={"type": "string", ...}) turns a function into a Tool."""
    return lambda fn: Tool(fn.__name__, description, params, fn)


def run_tool(tools, block):
    """Run one tool_use block and return its tool_result block. Errors go back to the model."""
    try:
        out = tools[block.name].fn(**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(out)}
    except Exception as e:  # the model sees the error and can try again
        return {"type": "tool_result", "tool_use_id": block.id,
                "content": f"{type(e).__name__}: {e}", "is_error": True}


# ---- Per-step record, run result, transcript ------------------------------------------

@dataclass
class Step:
    n: int
    input_tokens: int     # uncached input billed at the full rate
    cache_read: int       # input read from the prompt cache (0.1x price on Haiku 5.5)
    cache_write: int      # input written to the cache (1.25x price)
    output_tokens: int    # visible text, tool calls and thinking
    latency_s: float      # wall time of the model call
    cost: float           # USD, from the ledger's price table
    stop_reason: str
    tool_calls: list

    @property
    def context(self):    # everything the model read this step
        return self.input_tokens + self.cache_read + self.cache_write


@dataclass
class Result:
    answer: str
    stop: str             # end_turn, max_steps, max_cost, max_tokens or refusal
    steps: list = field(default_factory=list)

    @property
    def cost(self):
        return sum(s.cost for s in self.steps)


class Transcript:
    """Appends one JSON object per line: the task, each step's content and usage, the result."""
    def __init__(self, path):
        self.path = Path(path) if path else None

    def write(self, **record):
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")


class _TimedClient:
    """Wraps the SDK client so we time only messages.create, not the ledger's token count."""
    def __init__(self, cl):
        self.cl, self.messages, self.last = cl, self, 0.0

    def count_tokens(self, **kw):
        return self.cl.messages.count_tokens(**kw)

    def create(self, **kw):
        t0 = time.perf_counter()
        resp = self.cl.messages.create(**kw)
        self.last = time.perf_counter() - t0
        return resp


# ---- The loop ---------------------------------------------------------------------------

def run_agent(task, tools, system, *, part, model="claude-haiku-5-5", effort="medium",
              max_steps=12, max_cost=0.05, max_tokens=8000, cache=True, parallel=True,
              thinking="omitted", transcript=None, client=None, verbose=True):
    tools = {t.name: t for t in tools}
    request = dict(model=model, max_tokens=max_tokens, tools=[t.schema() for t in tools.values()],
                   system=[{"type": "text", "text": system}], output_config={"effort": effort},
                   thinking={"type": "adaptive", "display": thinking})
    if not parallel:  # at most one tool call per step
        request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    if cache:
        request["system"][0]["cache_control"] = {"type": "ephemeral"}  # caches tools + system
        request["cache_control"] = {"type": "ephemeral"}  # moves forward with the conversation
    cl = _TimedClient(client or ledger.client())
    log = Transcript(transcript)
    log.write(event="task", task=task, cache=cache, **request)

    messages = [{"role": "user", "content": task}]
    result = Result(answer="", stop="max_steps")
    for n in range(1, max_steps + 1):
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        resp = ledger.create(cl, part, note=f"agent step {n}", messages=messages, **request)
        u, calls = resp.usage, [b for b in resp.content if b.type == "tool_use"]
        step = Step(n, u.input_tokens, u.cache_read_input_tokens or 0,
                    u.cache_creation_input_tokens or 0, u.output_tokens, round(cl.last, 3),
                    ledger.cost_of(model, u), resp.stop_reason,
                    [f"{b.name}({json.dumps(b.input)})" for b in calls])
        result.steps.append(step)
        log.write(event="step", **step.__dict__,
                  content=[b.model_dump(exclude_none=True) for b in resp.content])
        if verbose:
            print(f"step {n:2d}  in {step.input_tokens:5,d}  cache read {step.cache_read:6,d}  "
                  f"write {step.cache_write:5,d}  out {step.output_tokens:4,d}  {step.latency_s:5.2f}s  "
                  f"${step.cost:.5f}  {' '.join(step.tool_calls) or resp.stop_reason}")

        messages.append({"role": "assistant", "content": resp.content})  # keep thinking blocks
        if resp.stop_reason != "tool_use":  # end_turn, max_tokens or refusal: the loop is over
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = [run_tool(tools, b) for b in calls]
        messages.append({"role": "user", "content": results})  # all results in one message
        log.write(event="tool_results", n=n, results=results)

    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps),
              cost=result.cost)
    return result
