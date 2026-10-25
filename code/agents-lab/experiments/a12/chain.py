"""The chain task: n dependent steps, each with a small chance of a silent mistake.

Every 3-digit code 000-999 is a record with a multiplier m and an offset d. The next code is the
last three digits of code * m + d. The agent opens records one at a time (each open is a tool
call) and must do the arithmetic itself. Every code exists, so a wrong step is silent: the agent
lands on a valid record and carries on down the wrong chain.
"""
import random
import re
import sys

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from harness import tool  # noqa: E402

SYSTEM = """You follow a chain of numbered records. Every code from 000 to 999 is a record with a
multiplier m and an offset d. The next code is the last three digits of (code x m + d), written with
three digits. Example: code 417 with m = 7 and d = 286 gives 417 x 7 + 286 = 3205, so the next code is 205.

Open records one at a time with open_record, in order, starting from the code you are given. The tool
tells you how many records you have opened. After the last record, compute the next code from it
(do not open it) and end your reply with one line: ANSWER: <three-digit code>"""

RESTATE = """

Before each open_record call, first copy the record you just opened as "Record <code>: m = <m>, d = <d>",
then write out the multiplication and the addition, then take the last three digits."""

SILENT = """

Work silently: write no text at all before or between tool calls. Call open_record with the next code
directly, doing the arithmetic in your head. Your final reply is only the ANSWER line."""


LEVELS = {"easy": (2, 9), "hard": (12, 99), "xhard": (101, 999)}  # range of the multiplier m


def record(seed, level, hop, code, redraw=0):
    """m and d of `code` at hop `hop`. Every hop draws fresh values, so a wrong turn almost never
    rejoins the right path (1 chance in 1,000 per step)."""
    rng = random.Random(f"{seed}-{level}-{hop}-{code}-{redraw}")
    lo, hi = LEVELS[level]
    return rng.randint(lo, hi), rng.randint(100, 999)


def nxt(code, m, d):
    return (code * m + d) % 1000


def make_task(seed, n, level="easy"):
    start = random.Random(seed * 7919 + n).randint(100, 999)
    path, recs = [start], []
    for hop in range(1, n + 1):
        for redraw in range(1000):  # the correct path never revisits a code, so it never looks like a cycle
            md = record(seed, level, hop, path[-1], redraw)
            if nxt(path[-1], *md) not in path:
                break
        recs.append(md)
        path.append(nxt(path[-1], *md))
    q = (f"Start at code {start:03d}. Open {n} records in total, following the chain, then give the "
         f"code that comes after the last record you opened.")
    return {"id": f"{level[0]}{seed}-n{n}", "seed": seed, "n": n, "level": level, "start": start, "question": q,
            "answer": f"{path[-1]:03d}", "path": path, "recs": recs}


class ChainState:
    def __init__(self, task):
        self.task, self.opened = task, []   # opened: (code, m, d) per hop

    def snapshot(self):
        return {"opened": [list(o) for o in self.opened]}

    def restore(self, d):
        self.opened = [tuple(o) for o in d["opened"]]


def make_tools(state):
    n = state.task["n"]

    @tool("Open the record with this three-digit code and see its multiplier and offset.",
          code={"type": "string", "description": "three digits, e.g. 042"})
    def open_record(code):
        if not re.fullmatch(r"\d{3}", code.strip()):
            raise ValueError("code must be exactly three digits, e.g. 042")
        if len(state.opened) >= n:
            return f"You have already opened all {n} records. Compute the final code and answer."
        c = int(code)
        hop, t = len(state.opened) + 1, state.task
        m, d = t["recs"][hop - 1] if c == t["path"][hop - 1] else record(t["seed"], t["level"], hop, c)
        state.opened.append((c, m, d))
        k = len(state.opened)
        msg = f"Record {c:03d}: m = {m}, d = {d}. Opened {k} of {n}."
        if k == n:
            msg += " That was the last record: compute the next code from it and give it as your answer."
        return msg
    return [open_record]


def final(answer):
    m = re.findall(r"ANSWER:\s*(\d{1,3})", answer or "")
    return f"{int(m[-1]):03d}" if m else None


def step_record(state, answer):
    """Per-step correctness, conditional on where the agent actually was: step i is right if the
    code it moved to (the next open, or the final answer) equals f(code it had opened)."""
    seq = [c for c, _, _ in state.opened]
    a = final(answer)
    moved = seq[1:] + ([int(a)] if a is not None else [])
    steps = [int(moved[i] == nxt(*state.opened[i])) for i in range(min(len(seq), len(moved)))]
    return {"opened": seq, "steps": steps, "start_ok": bool(seq) and seq[0] == state.task["start"]}
