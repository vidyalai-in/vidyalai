"""Self-correction strategies for Agentic AI Part 4, on one fixed set of 40 coding tasks.

For each (task, repeat) the model writes attempt 1 once; every strategy starts from that same
attempt, so differences between strategies are paired:

  single     attempt 1 as is
  critique   the model reviews its own code against the spec and revises (2 rounds, no tests)
  tests      the public tests run; failures go back to the model, which revises (up to 3 rounds)
  reflexion  on public-test failure the model writes a lesson, then retries with it (up to 3)
  resample   on public-test failure, draw a fresh attempt with no feedback (up to 3)
  oracle     like tests, but the feedback comes from the full hidden suite (an upper bound)
  tool       the Part 1 agent loop with a run_tests tool on the public suite (model decides)

Grading always uses the full (hidden) suite. Every API call goes through ledger.create.
"""

import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
import harness  # noqa: E402
from sandbox import run_tests  # noqa: E402
from tasks import TASKS  # noqa: E402

PART = "a04"
EXPECTED = json.loads((Path(__file__).with_name("expected.json")).read_text())
_lock = threading.Lock()

SYSTEM = ("You are an expert Python programmer. Implement the function the user describes, "
          "following every rule in its specification exactly. Use only the standard library. "
          "Reply with the complete function, plus any imports or helpers it needs, in a single "
          "```python code block.")

CRITIQUE = ("Review your code against the specification. Assume it could be either correct or "
            "incorrect. Check each rule and edge case carefully and report any real problems you "
            "find. Then give the final version of the complete function in a single ```python code "
            "block (unchanged if you found no real problems).")

REFLECT = ("Before writing any more code, reflect on this failure. In a few sentences, explain "
           "what went wrong and write a lesson to follow on your next attempt. Do not write code.")


def extract_code(text, entry=None):
    """The last fenced code block, preferring the last one that defines the entry function."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if entry:
        defining = [b for b in blocks if re.search(rf"^\s*def {entry}\(", b, re.M)]
        if defining:
            return defining[-1]
    return blocks[-1] if blocks else ""


def grade(t, code, public_only=False):
    """Run the tests; return (all_passed, failures as text lines)."""
    n = t["n_public"] if public_only else len(t["tests"])
    res = run_tests(code, t["id"], t["tests"][:n], EXPECTED[t["id"]][:n])
    fails = [f"- {r['call'][:200]}: expected {e[:200]}, got {r['got'][:200]}"
             for r, e in zip(res, EXPECTED[t["id"]][:n]) if not r["ok"]]
    return not fails, fails


class Model:
    """Calls the API through the ledger and logs every call (messages + usage) as JSONL."""

    def __init__(self, model, effort, log_path, thinking=True):
        self.model, self.effort, self.log_path = model, effort, Path(log_path)
        self.thinking = {"thinking": {"type": "adaptive"}} if thinking else {}
        self.cl = ledger.client().with_options(max_retries=10)  # back off on 429s

    def __call__(self, messages, note, tag):
        resp = ledger.create(self.cl, PART, note=note, model=self.model, max_tokens=16000,
                             system=SYSTEM, messages=messages, output_config={"effort": self.effort},
                             **self.thinking)
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        rec = dict(tag=tag, note=note, model=self.model, effort=self.effort, thinking=bool(self.thinking),
                   stop=resp.stop_reason, messages=messages, reply=text,
                   usage=dict(input=u.input_tokens, output=u.output_tokens,
                              cache_read=u.cache_read_input_tokens or 0,
                              cache_write=u.cache_creation_input_tokens or 0),
                   cost=ledger.cost_of(self.model, u))
        with _lock, self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return text, rec["usage"]["input"] + rec["usage"]["output"], rec["cost"]


def ask(t):
    return f"Implement this function:\n\n```python\n{t['spec']}\n```"


# ---- the strategies ---------------------------------------------------------------------

def critique(llm, t, first, tag, rounds=2):
    """Intrinsic self-correction: no tests, no labels. Returns the code after each round."""
    messages = [{"role": "user", "content": ask(t)}, {"role": "assistant", "content": first}]
    out, tokens, cost = [], 0, 0.0
    for r in range(rounds):
        messages.append({"role": "user", "content": CRITIQUE})
        reply, tk, c = llm(messages, f"critique r{r + 1}", tag)
        messages.append({"role": "assistant", "content": reply})
        out.append(dict(code=extract_code(reply, t["id"]) or extract_code(messages[-3]["content"], t["id"]),
                        reply=reply, tokens=tk, cost=c))
    return out


def with_tests(llm, t, first, tag, rounds=3, oracle=False):
    """Revise with test feedback: run the tests, send failures back, repeat until they pass."""
    messages = [{"role": "user", "content": ask(t)}, {"role": "assistant", "content": first}]
    code, calls = extract_code(first, t["id"]), []
    for r in range(rounds):
        ok, fails = grade(t, code, public_only=not oracle)
        if ok:
            break
        messages.append({"role": "user", "content":
                         f"I ran your code against the test suite. {len(fails)} test(s) failed:\n"
                         + "\n".join(fails) + "\n\nFix the code and reply with the complete "
                         "corrected function in a single ```python code block."})
        reply, tk, c = llm(messages, f"{'oracle' if oracle else 'tests'} r{r + 1}", tag)
        messages.append({"role": "assistant", "content": reply})
        code = extract_code(reply, t["id"]) or code
        calls.append(dict(reply=reply, tokens=tk, cost=c, code=code))
    return code, calls


def reflexion(llm, t, first, tag, trials=3):
    """Reflexion: on failure, write a verbal lesson; retry with the last attempt and all lessons."""
    code, lessons, calls = extract_code(first, t["id"]), [], []
    for r in range(trials):
        ok, fails = grade(t, code, public_only=True)
        if ok:
            break
        feedback = "Test results:\n" + "\n".join(fails)
        messages = [{"role": "user", "content": ask(t)},
                    {"role": "assistant", "content": f"```python\n{code}```"},
                    {"role": "user", "content": feedback + "\n\n" + REFLECT}]
        lesson, tk, c = llm(messages, f"reflexion reflect {r + 1}", tag)
        lessons.append(lesson.strip())
        calls.append(dict(reply=lesson, tokens=tk, cost=c, kind="reflect"))
        memory = "\n\n".join(f"Lesson {i + 1}: {x}" for i, x in enumerate(lessons))
        prompt = (ask(t) + f"\n\nYour previous attempt:\n```python\n{code}```\n{feedback}\n\n"
                  f"Lessons from your earlier attempts:\n{memory}\n\nWrite an improved implementation.")
        reply, tk, c = llm([{"role": "user", "content": prompt}], f"reflexion retry {r + 1}", tag)
        code = extract_code(reply, t["id"]) or code
        calls.append(dict(reply=reply, tokens=tk, cost=c, kind="retry", code=code))
    return code, calls


def resample(llm, t, first, tag, tries=3):
    """Verifier as a filter: on public-test failure, draw a fresh, independent attempt."""
    code, calls = extract_code(first, t["id"]), []
    for r in range(tries):
        ok, _ = grade(t, code, public_only=True)
        if ok:
            break
        reply, tk, c = llm([{"role": "user", "content": ask(t)}], f"resample {r + 1}", tag)
        code = extract_code(reply, t["id"]) or code
        calls.append(dict(reply=reply, tokens=tk, cost=c, code=code))
    return code, calls


def tool_agent(t, tag, model, effort, log_dir):
    """The Part 1 loop with one tool, run_tests, on the public suite. The model decides."""
    @harness.tool("Run your implementation against the public test suite. Pass the complete code "
                  "(function plus imports and helpers). Returns PASS or the failing tests.",
                  code={"type": "string", "description": "Complete Python source of the implementation"})
    def run_public_tests(code):
        last["code"] = code
        ok, fails = grade(t, code, public_only=True)
        return f"PASS: all {t['n_public']} public tests passed" if ok else \
            f"{len(fails)} of {t['n_public']} public tests failed:\n" + "\n".join(fails)

    last = {"code": ""}
    system = SYSTEM + (" You can check your code with the run_public_tests tool before answering. "
                       "End with your final code in a ```python block.")
    res = harness.run_agent(ask(t), [run_public_tests], system, part=PART, model=model,
                            effort=effort, max_steps=6, max_cost=0.05, cache=True,
                            transcript=Path(log_dir) / f"tool-{tag}.jsonl", verbose=False,
                            client=ledger.client().with_options(max_retries=10))
    tokens = sum(s.input_tokens + s.cache_read + s.cache_write + s.output_tokens for s in res.steps)
    return extract_code(res.answer, t["id"]) or last["code"], dict(steps=len(res.steps), tokens=tokens, cost=res.cost,
                                          tests_called=sum(len(s.tool_calls) for s in res.steps))
