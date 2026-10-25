"""Grader self-test: the untouched base must fail, the base plus the real fix must pass."""
import json, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tasklib import A08, TASKS, grade, load_tasks

def check(t):
    out = []
    for gold in (False, True):
        d = Path(tempfile.mkdtemp(dir=A08 / "tmp")); shutil.copytree(TASKS / t["id"] / "base", d, dirs_exist_ok=True)
        if gold:
            diff = next(json.loads(l)["diff"] for l in (A08/"candidates"/f"{t['repo']}.jsonl").read_text().splitlines()
                        if json.loads(l)["commit"] == t["commit_fix"])
            subprocess.run(["git", "apply", "-"], cwd=d, input=diff, text=True, check=True)
        out.append(grade(t, d)["resolved"]); shutil.rmtree(d)
    return t["id"], out

with ProcessPoolExecutor(6) as ex:
    for tid, (b, g) in ex.map(check, load_tasks()):
        print(f"{tid:30s} base={b} gold={g} {'OK' if (not b and g) else 'BAD'}")
