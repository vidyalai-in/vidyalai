"""Shared helpers for the a08 mini-benchmark: load tasks, run pytest, grade a workspace."""

import json
import re
import shutil
import subprocess
from pathlib import Path

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
PY = str(A08 / "venv/bin/python")
TASKS = A08 / "tasks"


def pythonpath(task, root):
    root = Path(root)
    return str(root / "src") if task["src_prefix"].startswith("src/") else str(root)


def run_pytest(task, root, files, timeout=300):
    """Run pytest on `files` in `root` (no network). Returns {node_id: PASSED|FAILED|ERROR|...}."""
    cmd = ["unshare", "-rn", PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "--timeout=30",
           "-o", "addopts=", "-W", "ignore", *files]
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout,
                           env={"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath(task, root),
                                "HOME": str(root)})
    except subprocess.TimeoutExpired:
        return {}
    res = {}
    for line in p.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED) (.+)$", line)
        if not m:
            continue
        status, rest = m.groups()
        if status in ("FAILED", "ERROR") and " - " in rest:
            rest = rest.rsplit(" - ", 1)[0] if rest.count(" - ") == 1 else rest.split(" - ")[0]
        res[rest.strip()] = status
    return res


def load_tasks():
    return [json.loads((d / "task.json").read_text()) for d in sorted(TASKS.iterdir())
            if (d / "task.json").exists()]


def grade(task, root):
    """Put the hidden test files back (discarding the agent's changes to them), run them, and check
    that every fail-to-pass test passes and no pass-to-pass test breaks."""
    root = Path(root)
    hidden = TASKS / task["id"] / "hidden"
    for f in hidden.rglob("*"):
        if f.is_file():
            dst = root / f.relative_to(hidden)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    res = run_pytest(task, root, task["test_files"])
    f2p_ok = sum(res.get(t) == "PASSED" for t in task["f2p"])
    p2p_bad = [t for t in task["p2p"] if res.get(t) != "PASSED"]
    return dict(resolved=bool(res) and f2p_ok == len(task["f2p"]) and not p2p_bad,
                f2p_pass=f2p_ok, f2p_total=len(task["f2p"]), p2p_fail=len(p2p_bad))
