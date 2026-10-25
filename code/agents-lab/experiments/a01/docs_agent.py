"""Part 1 experiment: a docs-folder question-answering agent built on lab/harness.py."""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from harness import tool, run_agent  # noqa: E402

ROOT = Path(__file__).parent / "sandbox"


def _safe(path):
    p = (ROOT / path).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"{path!r} is outside the docs folder")
    return p


@tool("List the files and folders inside a folder of the docs sandbox. Use '.' for the top level. "
      "Returns one entry per line: folders end with '/', files show their size in bytes. "
      "Use this to discover what exists before reading files.",
      path={"type": "string", "description": "Folder path relative to the sandbox root, e.g. '.' or 'docs/services'"})
def list_files(path):
    p = _safe(path)
    return "\n".join(f"{c.name}/" if c.is_dir() else f"{c.name}  ({c.stat().st_size} bytes)"
                     for c in sorted(p.iterdir()))


@tool("Read a whole text file (Markdown or CSV) from the docs sandbox and return its contents. "
      "Use the exact relative path shown by list_files or search, e.g. 'docs/teams/payments.md'.",
      path={"type": "string", "description": "File path relative to the sandbox root"})
def read_file(path):
    return _safe(path).read_text()


@tool("Search every file in the docs sandbox for a case-insensitive text or regular expression. "
      "Returns up to 30 matching lines as 'path:line_number: text'. Use it to find which file "
      "mentions a service, person, number or term; then read_file for full context.",
      pattern={"type": "string", "description": "Text or Python regular expression, e.g. 'port 7001' or 'depends on:.*billing'"})
def search(pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for f in sorted(ROOT.rglob("*")):
        if f.is_file():
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    return "\n".join(hits[:30]) or "no matches"


TOOLS = [list_files, read_file, search]

SYSTEM = """You answer questions about Larkspur Robotics, a fictional company, using its internal docs folder.
The folder holds Markdown pages under docs/ (services, teams, runbooks, policies, changelog) and CSV exports under data/.

How to work:
- Use the tools to find the facts. Do not guess and do not rely on outside knowledge.
- Questions may need several hops, for example from a service page to its owner team's page.
- For counts and totals over a CSV file, read the whole file and go through every row carefully.
- Stop calling tools as soon as you have the facts you need.

When you are done, reply with one or two sentences of explanation, then a last line in exactly this form:
ANSWER: <the answer>
If the docs do not contain the answer, write ANSWER: unknown"""


def answer_line(text):
    lines = [l for l in text.splitlines() if l.strip().upper().startswith("ANSWER:")]
    return lines[-1].split(":", 1)[1].strip() if lines else ""


def _norm(s):
    s = s.lower().replace("á", "a")
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def grade(task, text):
    """Correct if the ANSWER line matches one alternative from every group (word-bounded)."""
    ans = _norm(answer_line(text))
    def hit(alt):
        return re.search(r"(?<![\w.])" + re.escape(_norm(alt)) + r"(?![\w])", ans) is not None
    return bool(ans) and all(any(hit(a) for a in group) for group in task["must"])


if __name__ == "__main__":
    import json
    tasks = {t["id"]: t for t in json.loads((Path(__file__).parent / "tasks.json").read_text())}
    t = tasks[sys.argv[1] if len(sys.argv) > 1 else "t11"]
    r = run_agent(t["question"], TOOLS, SYSTEM, part="a01",
                  transcript=Path(__file__).parent / "runs" / f"demo-{t['id']}.jsonl")
    print(r.answer)
    print("correct:", grade(t, r.answer), "| stop:", r.stop, "| steps:", len(r.steps), f"| cost: ${r.cost:.5f}")
