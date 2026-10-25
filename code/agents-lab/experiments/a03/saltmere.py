"""The Saltmere Islands tools and grader for Part 3.

Four read-only tools over world.json (search, lookup, calculate, roads) and two stateful puzzle tools
(weigh, guess) that are live only in the puzzle tasks that need them. make_tools(task) builds a fresh
set per run, so a puzzle's use counter starts at zero every time.
"""
import ast
import difflib
import json
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from harness import Tool  # noqa: E402

HERE = Path(__file__).parent
WORLD = json.loads((HERE / "world.json").read_text())
RECORDS, ROADS = WORLD["records"], WORLD["roads"]
TASKS = json.loads((HERE / "tasks.json").read_text())

SYSTEM = """You answer questions about the Saltmere Islands, a fictional archipelago, and solve small
puzzles, using the tools provided. The records are the only source of truth: look facts up, never
guess them. Use calculate for arithmetic. The weigh and guess tools have a limited number of uses per
task, so make every use count. When you are finished, end your reply with a line in exactly this form:
ANSWER: <the answer>"""

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("round", "abs", "min", "max"):
        return {"round": round, "abs": abs, "min": min, "max": max}[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("only numbers, + - * / // % **, round, abs, min and max are allowed")


def search(query):
    q = query.lower().strip()
    hits = [f"{name} ({text.split('(')[1].split(')')[0]})" for name, text in RECORDS.items()
            if q and q in text.lower()]
    return "\n".join(hits[:12]) + (f"\n... {len(hits) - 12} more" if len(hits) > 12 else "") or "no matches"


def lookup(name):
    if name in RECORDS:
        return RECORDS[name]
    close = difflib.get_close_matches(name, RECORDS, n=3, cutoff=0.5)
    return f"No record named {name!r}." + (f" Closest names: {', '.join(close)}" if close else "")


def calculate(expression):
    value = _eval(ast.parse(expression, mode="eval").body)
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def roads(town):
    if town not in ROADS:
        return lookup(town)
    return "\n".join(f"{town} - {other}: {km} km" for other, km in sorted(ROADS[town].items()))


def make_tools(task):
    scale, lock = task.get("scale"), task.get("lock")
    used = {"weigh": 0, "guess": 0}

    def weigh(left, right):
        if not scale:
            raise ValueError("there is no scale in this task")
        if used["weigh"] >= scale["max"]:
            raise ValueError(f"no weighings left: the limit of {scale['max']} is used up")
        coins = left + right
        if len(set(coins)) != len(coins) or not all(1 <= c <= scale["n"] for c in coins):
            raise ValueError(f"each coin 1-{scale['n']} may appear at most once")
        used["weigh"] += 1
        w = lambda side: sum(10 + (scale["delta"] if c == scale["odd"] else 0) for c in side)  # noqa: E731
        res = ("balanced" if w(left) == w(right) else
               "left side heavier" if w(left) > w(right) else "right side heavier")
        return f"{res} (weighings left: {scale['max'] - used['weigh']})"

    def guess(code):
        if not lock:
            raise ValueError("there is no lock in this task")
        n = len(lock["code"])
        if not re.fullmatch(rf"\d{{{n}}}", code) or len(set(code)) != n:
            raise ValueError(f"a guess is {n} different digits")
        if used["guess"] >= lock["max"]:
            raise ValueError(f"no guesses left: the limit of {lock['max']} is used up")
        used["guess"] += 1
        bulls = sum(a == b for a, b in zip(code, lock["code"]))
        cows = len(set(code) & set(lock["code"])) - bulls
        return f"bulls: {bulls}, cows: {cows} (guesses left: {lock['max'] - used['guess']})"

    s = {"type": "string"}
    ints = {"type": "array", "items": {"type": "integer"}}
    return [
        Tool("search", "Search every Saltmere record for a case-insensitive text. Returns the names "
             "and types of up to 12 matching records.", {"query": {**s, "description": "Text to find, e.g. 'Mayor: Rosa Frome' or 'Headquarters: Penmoor'"}}, search),
        Tool("lookup", "Return the full record for one exact name: an island, town, person, company or "
             "ferry. Suggests close names if there is no exact match.", {"name": {**s, "description": "Exact record name, e.g. 'Port Wren'"}}, lookup),
        Tool("calculate", "Evaluate an arithmetic expression with + - * / // % ** and round(), abs(), "
             "min(), max().", {"expression": {**s, "description": "e.g. 'round(31788 / 22.3)'"}}, calculate),
        Tool("roads", "List the roads that leave a town, with their lengths in km.",
             {"town": {**s, "description": "Exact town name"}}, roads),
        Tool("weigh", "Puzzle tool: put coins on a balance scale and see which side is heavier. Limited "
             "uses; the task says how many.", {"left": {**ints, "description": "Coin numbers on the left pan"},
                                               "right": {**ints, "description": "Coin numbers on the right pan"}}, weigh),
        Tool("guess", "Puzzle tool: try a code on the lock and get bulls and cows back. "
             "Limited uses; the task says how many.", {"code": {**s, "description": "The code to try, as a string of digits, e.g. '5071'"}}, guess),
    ]


def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip().strip("*`. ") if m else None


def grade(task, text):
    a = final_answer(text)
    if a is None:
        return False
    if task["atype"] == "int":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", a)
        return bool(nums) and float(nums[0].replace(",", "")) == float(task["answer"])
    if task["atype"] == "coin":
        num, kind = task["answer"].split()
        nums = re.findall(r"\d+", a)
        return bool(nums) and nums[0] == num and kind[:4] in a.lower() and \
            ("heav" if kind == "lighter" else "ligh") not in a.lower()
    if task["atype"] == "code":
        return re.sub(r"\D", "", a) == task["answer"]
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()  # noqa: E731
    return norm(a) == norm(task["answer"])
