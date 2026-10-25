"""Harvest candidate tasks by "PR mirroring": undo a past bug-fix commit's source change on the
current HEAD and keep the tests that then fail (fail-to-pass) as the hidden tests.

For each repo, walk recent first-parent commits whose message looks like a fix and that touch both
source and test files. Reverse-apply only the source part of the diff to a clean copy of HEAD, run
the test files the commit touched, and record tests that pass at HEAD but fail after the revert.
Output: candidates/<repo>.jsonl
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
REPOS = A08 / "repos"
PY = str(A08 / "venv/bin/python")
OUT = A08 / "candidates"
FIX = re.compile(r"\b(fix|fixes|fixed|bug|issue|incorrect|wrong|handle|crash|regression)\b", re.I)

CFG = {  # repo -> (source prefix, tests prefix)
    "sqlparse": ("sqlparse/", "tests/"), "mistune": ("src/mistune/", "tests/"),
    "marshmallow": ("src/marshmallow/", "tests/"), "jmespath.py": ("jmespath/", "tests/"),
    "cachetools": ("src/cachetools/", "tests/"), "python-slugify": ("slugify/", "tests/"),
    "parse": ("parse/", "tests/"), "python-semver": ("src/semver/", "tests/"),
    "toolz": ("toolz/", "toolz/tests/"), "tomlkit": ("tomlkit/", "tests/"),
    "humanize": ("src/humanize/", "tests/"),
}


def git(repo, *args, inp=None):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, input=inp).stdout


def fresh_copy(repo, dest):
    """A clean copy of HEAD (tracked files only)."""
    a = subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=a.stdout, check=True)
    if repo.name == "humanize":
        (dest / "src/humanize/_version.py").write_text('__version__ = version = "4.99.0"\n')


def pythonpath(name, d):
    return str(d / "src") if (d / "src").is_dir() else str(d)


def run_tests(name, d, files, timeout=120):
    """Return {test_id: 'PASSED'|'FAILED'|'ERROR'} for the given test files."""
    cmd = ["unshare", "-rn", PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "--timeout=20",
           "-o", "addopts=", "-W", "ignore", *files]
    try:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout,
                           env={"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath(name, d), "HOME": str(d)})
    except subprocess.TimeoutExpired:
        return None
    res = {}
    for line in p.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR) (\S+)", line)
        if m:
            res[m.group(2)] = m.group(1)
    return res


def harvest(name, max_commits=700):
    repo = REPOS / name
    src, tst = CFG[name]
    log = git(repo, "log", "--first-parent", f"-{max_commits}", "--format=%H%x09%cs%x09%s")
    out = []
    baseline = {}
    tmp_root = A08 / "tmp"
    tmp_root.mkdir(exist_ok=True)
    for line in log.splitlines():
        h, date, subj = line.split("\t", 2)
        if not FIX.search(subj):
            continue
        files = git(repo, "diff", "--numstat", f"{h}^", h).splitlines()
        sfiles, tfiles, nlines = [], [], 0
        for f in files:
            parts = f.split("\t")
            if len(parts) != 3 or parts[0] == "-":
                continue
            a, r, path = parts
            if path.startswith(tst) and path.endswith(".py"):
                tfiles.append(path)
            elif path.startswith(src) and path.endswith(".py"):
                sfiles.append(path)
                nlines += int(a) + int(r)
        if not sfiles or not tfiles or len(sfiles) > 2 or nlines > 40:
            continue
        diff = git(repo, "diff", f"{h}^", h, "--", *sfiles)
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        try:
            fresh_copy(repo, d)
            tfiles_now = [t for t in tfiles if (d / t).exists()]
            if not tfiles_now:
                continue
            ap = subprocess.run(["git", "apply", "-R", "--check", "-"], cwd=d, input=diff,
                                capture_output=True, text=True)
            if ap.returncode != 0:
                continue
            key = tuple(sorted(tfiles_now))
            if key not in baseline:
                baseline[key] = run_tests(name, d, list(key))
            base = baseline[key]
            if not base:
                continue
            subprocess.run(["git", "apply", "-R", "-"], cwd=d, input=diff, capture_output=True, text=True)
            after = run_tests(name, d, list(key))
            if after is None:
                continue
            f2p = sorted(t for t, s in base.items() if s == "PASSED" and after.get(t) != "PASSED")
            p2p = sorted(t for t, s in base.items() if s == "PASSED" and after.get(t) == "PASSED")
            if f2p:
                out.append(dict(repo=name, commit=h, date=date, subject=subj, src_files=sfiles,
                                test_files=list(key), diff_lines=nlines, f2p=f2p, n_p2p=len(p2p),
                                diff=diff))
                print(f"{name} {h[:8]} {date} f2p={len(f2p)} p2p={len(p2p)} {subj[:70]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    return name, len(out)


if __name__ == "__main__":
    names = sys.argv[1:] or list(CFG)
    with ProcessPoolExecutor(6) as ex:
        for name, n in ex.map(harvest, names):
            print("DONE", name, n, flush=True)
