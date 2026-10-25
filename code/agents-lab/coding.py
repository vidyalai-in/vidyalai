"""Part 8: a small coding agent on top of the Part 1 harness and the Part 3 loop.

The agent works on a throwaway copy of a Python repository and gets tools from this menu:

  grep         regex search over the repo, with file and line numbers
  glob         find files by name pattern
  list_dir     list one directory (the navigation tool when search is switched off)
  read_file    read a range of lines, numbered
  str_replace  replace one exact, unique snippet in a file
  create_file  create a new file (never overwrites)
  write_file   write a whole file (the whole-file-rewrite edit tool)
  run_tests    run pytest with the given arguments, sandboxed, no network

Every path is relative to the repo root and may not leave it. run_tests runs the model's code in
a new network namespace (unshare -rn) with CPU, memory and wall-clock limits.

harness.py, react.py and ledger.py are imported, not changed.
"""

import ast
import re
import shlex
import resource
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import tool  # noqa: E402

PY = "/home/amit/.cache/vidyalai-scratch/a08/venv/bin/python"  # the task venv: pytest + test deps
MAX_LINES = 400        # read_file returns at most this many lines per call
MAX_OUT = 6000         # characters of pytest output returned to the model


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))  # 2 GB address space
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def make_tools(root, pythonpath, kit):
    """Build the tools for one run. `kit` is a set of tool names to include."""
    root = Path(root).resolve()

    def resolve(path):
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"{path} is outside the repository")
        return p

    @tool("Search file contents with an extended regular expression (grep -E). Returns up to 50 "
          "matching lines as path:line: text. `path` is a file or directory relative to the repo root "
          "('.' for everything).", pattern={"type": "string"}, path={"type": "string"})
    def grep(pattern, path):
        p = subprocess.run(["grep", "-rnIE", "--exclude-dir=.git", "--", pattern, str(resolve(path))],
                           capture_output=True, text=True, timeout=30)
        lines = [l.replace(str(root) + "/", "", 1)[:240] for l in p.stdout.splitlines()]
        if not lines:
            return "No matches." + (f" ({p.stderr.strip()[:200]})" if p.stderr else "")
        more = f"\n... {len(lines) - 50} more matches not shown; narrow the pattern or path." if len(lines) > 50 else ""
        return "\n".join(lines[:50]) + more

    @tool("Find files whose path matches a glob pattern relative to the repo root, e.g. '**/*.py' or "
          "'src/**/fields*.py'. Returns up to 100 paths.", pattern={"type": "string"})
    def glob(pattern):
        hits = sorted(str(p.relative_to(root)) for p in root.glob(pattern)
                      if p.is_file() and ".git" not in p.parts)
        if not hits:
            return "No files match."
        return "\n".join(hits[:100]) + (f"\n... {len(hits) - 100} more" if len(hits) > 100 else "")

    @tool("List the files and subdirectories of one directory, relative to the repo root ('.' for the "
          "root). Directories end with '/'.", path={"type": "string"})
    def list_dir(path):
        d = resolve(path)
        items = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        return "\n".join(p.name + ("/" if p.is_dir() else "") for p in items if p.name != ".git") or "(empty)"

    @tool(f"Read lines start_line..end_line (1-based, inclusive) of a text file, with line numbers. At "
          f"most {MAX_LINES} lines per call; use end_line -1 for the end of the file.",
          path={"type": "string"}, start_line={"type": "integer"}, end_line={"type": "integer"})
    def read_file(path, start_line, end_line):
        lines = resolve(path).read_text().splitlines()
        start = max(1, start_line)
        end = len(lines) if end_line < 0 else min(end_line, len(lines))
        end = min(end, start + MAX_LINES - 1)
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
        return f"{path} (lines {start}-{end} of {len(lines)})\n{body}"

    @tool("Replace one exact snippet in a file. old_str must match the file exactly, including "
          "indentation, and must occur exactly once; include enough surrounding lines to make it "
          "unique. new_str replaces it.", path={"type": "string"}, old_str={"type": "string"},
          new_str={"type": "string"})
    def str_replace(path, old_str, new_str):
        p = resolve(path)
        text = p.read_text()
        n = text.count(old_str)
        if n != 1:
            raise ValueError(f"old_str occurs {n} times in {path}; it must occur exactly once."
                             + (" Check whitespace and indentation, or read the file again." if n == 0 else
                                " Add surrounding lines to make it unique."))
        p.write_text(text.replace(old_str, new_str, 1))
        start = text[:text.index(old_str)].count("\n") + 1
        new_lines = p.read_text().splitlines()
        lo, hi = max(1, start - 3), min(len(new_lines), start + new_str.count("\n") + 3)
        return f"Edited {path}. Lines {lo}-{hi} now read:\n" + "\n".join(
            f"{i:5d}  {new_lines[i - 1]}" for i in range(lo, hi + 1))

    @tool("Create a new file with the given content (for example a test that reproduces the bug). "
          "Fails if the file already exists.", path={"type": "string"}, content={"type": "string"})
    def create_file(path, content):
        p = resolve(path)
        if p.exists():
            raise FileExistsError(f"{path} already exists; edit it instead")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Created {path} ({content.count(chr(10)) + 1} lines)."

    @tool("Write a file in full, replacing its whole content (or creating it). To change an existing "
          "file, read it first and send back the complete new version.",
          path={"type": "string"}, content={"type": "string"})
    def write_file(path, content):
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = p.read_text() if p.exists() else None
        p.write_text(content)
        return f"Wrote {path}: {content.count(chr(10)) + 1} lines" + (
            f" (was {old.count(chr(10)) + 1})." if old is not None else " (new file).")

    @tool("Run pytest in the repo root with the given arguments, e.g. 'tests/test_foo.py -k parse -x'. "
          "No network access. Returns the end of the output.", args={"type": "string"})
    def run_tests(args):
        try:
            argv = shlex.split(args)
        except ValueError as e:
            raise ValueError(f"could not split the arguments: {e}")
        for a in argv:
            if not a.startswith("-"):
                resolve(a.split("::")[0])  # test paths must stay inside the repo
        try:
            p = subprocess.run(["unshare", "-rn", PY, "-m", "pytest", "--no-header", "-p", "no:cacheprovider",
                                "--timeout=30", "-o", "addopts=", *argv], cwd=root, capture_output=True,
                               text=True, timeout=180, preexec_fn=_limits,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath, "HOME": str(root)})
            out = (p.stdout + p.stderr).replace(str(root) + "/", "")
        except subprocess.TimeoutExpired:
            return "pytest timed out after 180 s."
        return out if len(out) <= MAX_OUT else f"... ({len(out) - MAX_OUT} characters cut)\n" + out[-MAX_OUT:]

    menu = dict(grep=grep, glob=glob, list_dir=list_dir, read_file=read_file, str_replace=str_replace,
                create_file=create_file, write_file=write_file, run_tests=run_tests)
    return [menu[n] for n in kit]


# ---- A repo map: every source file with its classes and functions -----------------------------

def repo_map(root, src_prefix, budget_chars=8000):
    """Aider-style map, simplified: each source file, then its top-level classes (with method names)
    and functions, with line numbers. Trimmed to about budget_chars (~2,000 tokens)."""
    root = Path(root)
    out = []
    for f in sorted((root / src_prefix).rglob("*.py")):
        rel = f.relative_to(root)
        if "tests" in rel.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        names = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                meths = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and (not n.name.startswith("_") or n.name == "__init__")]
                names.append(f"class {node.name}:{node.lineno}" + (f" ({', '.join(meths[:12])})" if meths else ""))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(f"def {node.name}:{node.lineno}")
        out.append(f"{rel}: " + ("; ".join(names) if names else "(no top-level definitions)"))
    text = "\n".join(out)
    if len(text) > budget_chars:
        text = text[:budget_chars].rsplit("\n", 1)[0] + "\n... (map trimmed)"
    return text


ARMS = {  # tool kit and whether the task message carries a repo map
    "full": (["grep", "glob", "read_file", "str_replace", "create_file", "run_tests"], False),
    "no_tests": (["grep", "glob", "read_file", "str_replace", "create_file"], False),
    "rewrite": (["grep", "glob", "read_file", "write_file", "run_tests"], False),
    "no_search": (["list_dir", "read_file", "str_replace", "create_file", "run_tests"], False),
    "map": (["grep", "glob", "read_file", "str_replace", "create_file", "run_tests"], True),
    "map_no_search": (["list_dir", "read_file", "str_replace", "create_file", "run_tests"], True),
    "title_only": (["grep", "glob", "read_file", "str_replace", "create_file", "run_tests"], False),
}

SYSTEM = """You are a software engineer fixing a bug in a Python repository. You work through tools on a
checkout of the repository; every path is relative to the repository root. A user has filed the issue
below. Find the cause in the library's source code and fix it there, with the smallest change that
resolves the issue without breaking other behaviour. Hidden tests will check your fix, and any changes
you make to existing test files are discarded before they run. Do not ask questions: nobody will answer.
When you are done, reply with a two- or three-sentence summary of the cause and the fix."""

NO_RUN = "\nYou cannot run code or tests in this environment; reason from the source."


def task_message(task, root, arm):
    kit, with_map = ARMS[arm]
    issue = task["issue"].strip()
    if arm == "title_only":  # a terse report: the issue's first line, no reproduction
        issue = issue.splitlines()[0]
    msg = f"Repository: {task['repo']}\n\n<issue>\n{issue}\n</issue>"
    if "run_tests" in kit:
        msg += f"\n\nThe existing tests live under {task['tests_prefix']}; run_tests runs pytest."
    if with_map:
        msg += f"\n\nRepository map (source files, their classes and functions, line numbers):\n" \
               f"{repo_map(root, task['src_prefix'])}"
    return msg
