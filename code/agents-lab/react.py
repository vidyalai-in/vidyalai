"""Part 3: three ways to make an agent reason, on top of the Part 1 harness.

  run_loop          the harness loop with the thinking setting exposed (adaptive or disabled)
  REACT_RULES       ReAct-style prompting: a visible "Thought:" before every action
  plan_and_execute  a planner writes the whole plan up front, an executor runs it,
                    and the planner gets one chance to replan if the executor gives up

harness.py is imported, not changed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Result, Step, Transcript, _TimedClient, run_tool  # noqa: E402

ADAPTIVE = {"type": "adaptive", "display": "summarized"}   # model decides when and how much to think
DISABLED = {"type": "disabled"}                            # no thinking blocks at all (Haiku 5.5, effort <= high)

REACT_RULES = """
Work in the ReAct style. Before every tool call, write a short paragraph that starts with "Thought:"
and says what you know so far, what is still missing, and why the next action is the right one.
After each tool result, start your next message with a Thought about what the result means."""


@dataclass
class TStep(Step):
    thinking_tokens: int = 0  # billed output tokens spent on thinking (usage.output_tokens_details)


def run_loop(task, tools, system, *, part, model="claude-haiku-5-5", effort="medium",
             thinking=ADAPTIVE, max_steps=20, max_cost=0.05, max_tokens=16000,
             messages=None, transcript=None, client=None, verbose=False, timeout=None):
    """Part 1's loop, unchanged in shape; `thinking` and the starting `messages` are parameters.
    Pass `timeout` (seconds) to allow max_tokens above about 21,000 without streaming."""
    tools = {t.name: t for t in tools}
    request = dict(model=model, max_tokens=max_tokens, tools=[t.schema() for t in tools.values()],
                   system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, thinking=thinking,
                   cache_control={"type": "ephemeral"})
    if timeout:
        request["timeout"] = timeout  # an SDK option, not sent to the API
    cl = _TimedClient(client or ledger.client())
    log = Transcript(transcript)
    log.write(event="task", task=task, **request)

    messages = messages or [{"role": "user", "content": task}]
    result = Result(answer="", stop="max_steps")
    for n in range(1, max_steps + 1):
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        resp = ledger.create(cl, part, note=f"a03 step {n}", messages=messages, **request)
        u, calls = resp.usage, [b for b in resp.content if b.type == "tool_use"]
        details = getattr(u, "output_tokens_details", None)
        step = TStep(n, u.input_tokens, u.cache_read_input_tokens or 0,
                     u.cache_creation_input_tokens or 0, u.output_tokens, round(cl.last, 3),
                     ledger.cost_of(model, u), resp.stop_reason,
                     [f"{b.name}({json.dumps(b.input)})" for b in calls],
                     getattr(details, "thinking_tokens", 0) or 0)
        result.steps.append(step)
        log.write(event="step", **step.__dict__,
                  content=[b.model_dump(exclude_none=True) for b in resp.content])
        if verbose:
            print(f"step {n:2d} out {step.output_tokens:5d} (thinking {step.thinking_tokens:5d}) "
                  f"{step.latency_s:5.2f}s ${step.cost:.5f} {' '.join(step.tool_calls) or resp.stop_reason}")

        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = [run_tool(tools, b) for b in calls]
        messages.append({"role": "user", "content": results})
        log.write(event="tool_results", n=n, results=results)

    result.messages = messages
    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps),
              cost=result.cost)
    return result


# ---- plan-and-execute -----------------------------------------------------------------------

PLANNER = """You are the planner for a tool-using agent. Read the task and the tool list, then write a
numbered plan that an executor will follow. You cannot call tools yourself. Name the tool and the
input for each step. Where an input depends on an earlier result, refer to it as #E1, #E2 and so on,
and say how to use it. Where the next step depends on what a tool returns, write the rule for each
case. End with the step that computes the final answer. Write only the plan."""

EXECUTOR = """
You are the executor. A planner has written a plan for this task. Follow it step by step with the
tools, filling in each #E reference from the actual results. If a result shows the plan cannot work
(a name is missing, a branch was not foreseen, a limit would be exceeded), stop and reply with one
line "REPLAN: <what went wrong>" instead of an answer."""


def _plan(task, tools, request_tools, cl, *, part, model, effort, note, extra=""):
    resp = ledger.create(cl, part, note=note, model=model, max_tokens=4000,
                         system=[{"type": "text", "text": PLANNER, "cache_control": {"type": "ephemeral"}}],
                         tools=request_tools, tool_choice={"type": "none"},
                         output_config={"effort": effort}, thinking=DISABLED,
                         messages=[{"role": "user", "content": f"Task: {task}{extra}"}])
    plan = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    step = TStep(0, u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                 u.output_tokens, round(cl.last, 3), ledger.cost_of(model, u), resp.stop_reason,
                 [], 0)
    return plan, step


def _observations(messages):
    """The tool calls and results an executor made, as plain text for the replanner."""
    lines, calls = [], {}
    for m in messages:
        for b in m["content"] if isinstance(m["content"], list) else []:
            if getattr(b, "type", None) == "tool_use":
                calls[b.id] = f"{b.name}({json.dumps(b.input)})"
            elif isinstance(b, dict) and b.get("type") == "tool_result":
                lines.append(f"{calls.get(b['tool_use_id'], '?')} -> {b['content']}")
    return "\n".join(lines)


def plan_and_execute(task, tools, system, *, part, model="claude-haiku-5-5", effort="medium",
                     replan=True, max_steps=20, max_cost=0.05, transcript=None, client=None):
    cl = _TimedClient(client or ledger.client())
    log = Transcript(transcript)
    schemas = [t.schema() for t in tools]
    plan, pstep = _plan(task, tools, schemas, cl, part=part, model=model, effort=effort, note="a03 plan")
    log.write(event="plan", plan=plan, **pstep.__dict__)
    first = run_loop(f"Task: {task}\n\nPlan:\n{plan}", tools, system + EXECUTOR, part=part, model=model,
                     effort=effort, thinking=DISABLED, max_steps=max_steps, max_cost=max_cost,
                     max_tokens=4000, transcript=transcript, client=client)
    steps, result = [pstep] + first.steps, first
    if replan and ("REPLAN" in first.answer or "ANSWER:" not in first.answer):
        seen = _observations(first.messages)
        extra = (f"\n\nA first plan was tried:\n{plan}\n\nThe executor reported: "
                 f"{first.answer.strip() or first.stop}\n\nTool results so far (tools keep their state, "
                 f"and uses already spent stay spent):\n{seen}\n\nWrite a new plan for the remaining work.")
        plan2, pstep2 = _plan(task, tools, schemas, cl, part=part, model=model, effort=effort,
                              note="a03 replan", extra=extra)
        log.write(event="replan", plan=plan2, **pstep2.__dict__)
        second = run_loop(f"Task: {task}\n\nResults so far:\n{seen}\n\nPlan:\n{plan2}", tools,
                          system + EXECUTOR, part=part, model=model, effort=effort, thinking=DISABLED,
                          max_steps=max_steps, max_cost=max_cost, max_tokens=4000,
                          transcript=transcript, client=client)
        steps += [pstep2] + second.steps
        result = second
    out = Result(answer=result.answer, stop=result.stop, steps=steps)
    out.replanned = len(steps) > len(first.steps) + 1
    return out
