"""Build the mini-benchmark from the curated list in tasks_spec.json.

For each task: a clean copy of the repo at HEAD, with the fix commit's source change undone
("PR mirroring"), the fail-to-pass tests deleted from the visible test files, and agent instruction
files (AGENTS.md, CLAUDE.md) removed. The original test files are kept aside as the hidden tests.
Validation: gold (HEAD) passes every F2P and P2P test twice; the buggy base fails every F2P test.
"""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasklib import A08, TASKS, run_pytest  # noqa: E402
from harvest import CFG, fresh_copy  # noqa: E402

REPOS = A08 / "repos"


def remove_tests(path, names):
    """Delete the test functions/methods named in `names` (tuples like ('TestC', 'test_m') or
    ('test_f',)) from a test file, decorators included."""
    src = path.read_text()
    tree = ast.parse(src)
    spans = []

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + (node.name,))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and prefix + (node.name,) in names:
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                spans.append((start, node.end_lineno, node.col_offset))

    visit(tree.body, ())
    lines = src.splitlines(keepends=True)
    for start, end, col in sorted(spans, reverse=True):
        lines[start - 1:end] = []
    out = "".join(lines)
    try:
        ast.parse(out)
    except SyntaxError:  # a class left with no body
        lines = src.splitlines(keepends=True)
        for start, end, col in sorted(spans, reverse=True):
            lines[start - 1:end] = [" " * col + "pass\n"]
        out = "".join(lines)
    path.write_text(out)
    return len(spans)


def build(spec):
    tid, name = spec["id"], spec["repo"]
    repo = REPOS / name
    src_prefix, tests_prefix = CFG[name]
    cand = next(json.loads(l) for l in (A08 / "candidates" / f"{name}.jsonl").read_text().splitlines()
                if json.loads(l)["commit"].startswith(spec["commit"]))
    tdir = TASKS / tid
    shutil.rmtree(tdir, ignore_errors=True)
    (tdir / "hidden").mkdir(parents=True)
    task = dict(id=tid, repo=name, commit_fix=cand["commit"], fix_date=cand["date"],
                fix_subject=cand["subject"], head=subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(), src_prefix=src_prefix,
                tests_prefix=tests_prefix, src_files=cand["src_files"], test_files=cand["test_files"],
                gold_diff_lines=cand["diff_lines"], issue=spec["issue"])
    gold = Path(tempfile.mkdtemp(dir=A08 / "tmp"))
    try:
        fresh_copy(repo, gold)
        task0 = dict(task)
        g1 = run_pytest(task0, gold, task["test_files"])
        g2 = run_pytest(task0, gold, task["test_files"])
        base = tdir / "base"
        base.mkdir()
        fresh_copy(repo, base)
        for f in ("AGENTS.md", "CLAUDE.md"):
            (base / f).unlink(missing_ok=True)
        subprocess.run(["git", "apply", "-R", "-"], cwd=base, input=cand["diff"], text=True, check=True)
        b = run_pytest(task0, base, task["test_files"])
        stable = {t for t in g1 if g1[t] == "PASSED" and g2.get(t) == "PASSED"
                  and "test_invalid_datetime_deserialization[" not in t}
        f2p = sorted(t for t in stable if b.get(t) != "PASSED")
        p2p = sorted(t for t in stable if b.get(t) == "PASSED")
        task.update(f2p=f2p, p2p=p2p)
        for tf in task["test_files"]:  # hidden copies of the original test files
            dst = tdir / "hidden" / tf
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gold / tf, dst)
        names = {}
        for t in f2p:
            path, *parts = t.split("[")[0].split("::")
            names.setdefault(path, set()).add(tuple(parts))
        removed = sum(remove_tests(base / p, ns) for p, ns in names.items())
        # the buggy base, with hidden tests restored, must fail every F2P test
        chk = Path(tempfile.mkdtemp(dir=A08 / "tmp"))
        shutil.copytree(base, chk, dirs_exist_ok=True)
        for tf in task["test_files"]:
            shutil.copy2(tdir / "hidden" / tf, chk / tf)
        b2 = run_pytest(task0, chk, task["test_files"])
        shutil.rmtree(chk)
        task["valid"] = bool(f2p) and all(b2.get(t) != "PASSED" for t in f2p) and \
            all(b2.get(t) == "PASSED" for t in p2p)
        task["removed_test_functions"] = removed
        (tdir / "task.json").write_text(json.dumps(task, indent=1))
        return tid, len(f2p), len(p2p), removed, task["valid"]
    finally:
        shutil.rmtree(gold, ignore_errors=True)


if __name__ == "__main__":
    specs = json.loads((A08 / "tasks_spec.json").read_text())
    if len(sys.argv) > 1:
        specs = [s for s in specs if s["id"] in sys.argv[1:]]
    (A08 / "tmp").mkdir(exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(build, specs):
            print(*r, flush=True)
