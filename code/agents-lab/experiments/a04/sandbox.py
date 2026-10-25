"""Run model-written code against unit tests in a throwaway directory, with no network.

Each call writes the candidate code and the test calls into a fresh temp dir under
a04/sandbox/, then runs `unshare -rn python -I child.py` there (a new network namespace with
no interfaces, so no network), with a 1 GB address-space limit, 10 s of CPU and a 15 s wall
timeout. The child prints one JSON line per test.
"""

import json
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SANDBOX = Path(__file__).with_name("sandbox")
PY = "/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python"

CHILD = r'''
import json, sys
spec = json.load(open("job.json"))
out = []
ns = {}
try:
    exec(compile(spec["code"], "solution.py", "exec"), ns)
    load_error = None
except BaseException as e:
    load_error = f"{type(e).__name__}: {e}"
for call, expected in zip(spec["calls"], spec["expected"]):
    if load_error:
        out.append({"call": call, "ok": False, "got": "code failed to load: " + load_error}); continue
    try:
        got = eval(spec["entry"] + call, ns)
        g = repr(got)
        if expected is None:
            ok = None
        elif expected.startswith("raises "):
            ok = False
        else:
            want = eval(expected, {})
            ok = got == want and type(got) is type(want)
    except BaseException as e:
        names = [c.__name__ for c in type(e).__mro__]
        ok = expected is not None and expected.startswith("raises ") and expected.split()[1] in names
        g = f"raises {type(e).__name__}" if expected is None else f"raises {type(e).__name__}: {str(e)[:200]}"
    out.append({"call": spec["entry"] + call, "ok": ok, "got": g})
print(json.dumps(out))
'''


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_tests(code, entry, calls, expected, timeout=15):
    """Return a list of {call, ok, got} dicts, one per test call."""
    SANDBOX.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=SANDBOX))
    try:
        (d / "child.py").write_text(CHILD)
        (d / "job.json").write_text(json.dumps({"code": code, "entry": entry, "calls": calls,
                                                "expected": expected}))
        p = subprocess.run(["unshare", "-rn", PY, "-I", "child.py"], cwd=d, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            err = (p.stderr or "no output").strip().splitlines()[-1:] or ["no output"]
            return [{"call": c, "ok": False, "got": f"crashed: {err[0][:200]}"} for c in calls]
    except subprocess.TimeoutExpired:
        return [{"call": c, "ok": False, "got": f"timed out after {timeout} s"} for c in calls]
    finally:
        shutil.rmtree(d, ignore_errors=True)
