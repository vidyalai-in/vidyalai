"""Verifiers for code and math benchmarks (Measuring LLMs Part 8, "Code and math benchmarks").

Code: build a HumanEval program from a prompt and a model reply, run it against a test suite in a
throwaway directory with no network, and estimate pass@k with the unbiased estimator of Chen et al.
(2021, arXiv:2107.03374). Math: pull the last \\boxed{...} out of a reply and grade it three ways:
exact string match, the MATH repository's string normalization (Hendrycks et al. 2021,
modeling/math_equivalence.py), and symbolic equivalence with Hugging Face's math-verify.

Sandbox: each program runs as a child process inside `unshare --user --map-current-user --net`
(a new user and network namespace: no network, same unprivileged uid), with a wall-clock timeout,
CPU and file-size rlimits, one BLAS thread and a reliability guard that disables process, kill and
file-deletion functions (after human-eval's execution.py). Callers run the parent itself under
`systemd-run --user --scope -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=64`.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from math import comb
from pathlib import Path

import numpy as np

SANDBOX = Path("/home/amit/.cache/vidyalai-scratch/e08/sandbox")
PYTHON = "/home/amit/.cache/vidyalai-scratch/evals/.venv/bin/python"
UNSHARE = ["unshare", "--user", "--map-current-user", "--net"]

# ------------------------------------------------------------------ pass@k ----------------------


def pass_at_k(n, c, k):
    """Unbiased P(at least one of k samples passes), from c passes in n samples (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_table(counts, n, ks=(1, 5, 10), n_boot=10_000, seed=0):
    """counts: passes per problem (each out of n). Returns {k: (mean, lo, hi)} with a percentile
    bootstrap over problems (the problems are the sample, so they are what gets resampled)."""
    counts = np.asarray(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), (n_boot, len(counts)))
    out = {}
    for k in ks:
        per = np.array([pass_at_k(n, int(c), k) for c in counts])
        boots = per[idx].mean(1)
        out[k] = (float(per.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return out


# ------------------------------------------------------------------ code ------------------------

FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(reply, entry_point):
    """The first fenced block that defines `entry_point`, else the longest block, else the reply."""
    blocks = FENCE.findall(reply)
    if not blocks:
        # an unterminated block (reply cut off): take everything after the opening fence
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", reply, re.S)
        return m.group(1) if m else reply
    for b in blocks:
        if re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", b, re.M):
            return b
    return max(blocks, key=len)


def build_program(prompt, code):
    """The task prompt first (its imports and helper functions), then the model's code, whose
    definition of the entry point replaces the prompt's docstring-only stub."""
    return prompt + "\n    pass\n\n" + code + "\n"


GUARD = r'''
import os as _os, resource as _r, faulthandler as _fh
_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))
_r.setrlimit(_r.RLIMIT_FSIZE, (1 << 20, 1 << 20))
_fh.disable()
import numpy as _np  # imported before the guard: its import calls os.putenv
import builtins as _b, shutil as _sh, subprocess as _sp
for _n in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork",
           "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod",
           "fchown", "chmod", "chown", "chroot", "lchown", "execv", "execve", "spawnv"):
    if hasattr(_os, _n):
        setattr(_os, _n, None)
_sh.rmtree = _sh.move = _sh.chown = None
_sp.Popen = None
_b.exit = _b.quit = None
'''


def run_program(src, timeout=10.0, cpu=None):
    """Run `src` in a fresh temp dir under SANDBOX with no network. Returns (status, detail):
    status is "pass", "fail" (non-zero exit, e.g. AssertionError), or "timeout"."""
    cpu = cpu or int(timeout) + 1
    SANDBOX.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SANDBOX) as d:
        path = Path(d) / "prog.py"
        path.write_text(GUARD.format(cpu=cpu) + "\n" + src)
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "HOME": d}
        try:
            p = subprocess.run(UNSHARE + [PYTHON, "-I", str(path)], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        if p.returncode == 0:
            return "pass", ""
        err = (p.stderr or "").strip().splitlines()
        return "fail", (err[-1] if err else f"exit {p.returncode}")[:300]


def humaneval_check(task, code, timeout=10.0):
    """Run one completion against one HumanEval-format task (prompt, test, entry_point)."""
    src = build_program(task["prompt"], code) + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    return run_program(src, timeout=timeout)


# ------------------------------------------------------------------ math ------------------------


def last_boxed(text):
    """Contents of the last \\boxed{...} (or \\fbox{...}) with balanced braces, else None."""
    if text is None:
        return None
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    i = starts[-1]
    depth = 1
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[starts[-1]:i - 1].strip() if depth == 0 else None


def _after_eq(a):
    """'x = 9' -> '9', 'p(8) = \\frac{3}{56}' -> '\\frac{3}{56}' (the right-hand side of the last '=')."""
    a = a.strip().rstrip(".").strip()
    return a.rsplit("=", 1)[-1].strip() if "=" in a else a


def _unlabel(a):
    return re.sub(r"^(final\s+)?answer\s*[:=]?\s*", "", a.strip(), flags=re.I).strip().strip("$ ").rstrip(".").strip("$ ")


def extract_answer(text):
    """(answer, rule). In order: the last \\boxed{}; a final line that is all bold ("**Answer: 13**");
    the last $...$ or $$...$$ span of the final paragraph; the last bold span; else (None, "none").
    For the last three, only the right-hand side of a trailing "=" is kept."""
    b = last_boxed(text)
    if b is not None:
        return b, "boxed"
    text = (text or "").strip()
    last_line = text.splitlines()[-1].strip() if text else ""
    m = re.fullmatch(r"\*\*(.+?)\*\*\.?", last_line)
    if m:
        return _after_eq(_unlabel(m.group(1))), "bold line"
    last_par = text.split("\n\n")[-1]
    spans = re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$", last_par, re.S)
    if spans:
        return _after_eq([x or y for x, y in spans][-1]), "math span"
    bold = re.findall(r"\*\*(.+?)\*\*", text, re.S)
    if bold:
        return _after_eq(_unlabel(bold[-1])), "bold"
    return None, "none"


def grade_exact(pred, gold):
    return pred is not None and pred.strip() == gold.strip()


# The next five functions follow modeling/math_equivalence.py in github.com/hendrycks/math.

def _fix_fracs(s):
    subs = s.split("\\frac")
    new = subs[0]
    for sub in subs[1:]:
        new += "\\frac"
        if sub and sub[0] == "{":
            new += sub
        else:
            if len(sub) < 2:
                return s
            a, b = sub[0], sub[1]
            if b != "{":
                new += "{" + a + "}{" + b + "}" + sub[2:]
            else:
                new += "{" + a + "}" + b + sub[2:]
    return new


def _fix_a_slash_b(s):
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        a, b = int(a), int(b)
        assert s == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return s


def _remove_right_units(s):
    if "\\text{ " in s:
        return s.split("\\text{ ")[0]
    return s


def _fix_sqrt(s):
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    new = parts[0]
    for p in parts[1:]:
        new += ("\\sqrt{" + p[0] + "}" + p[1:]) if p and p[0] != "{" else ("\\sqrt" + p)
    return new


def strip_string(s):
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    return _fix_a_slash_b(s)


def grade_normalized(pred, gold):
    if pred is None:
        return False
    try:
        return strip_string(pred) == strip_string(gold)
    except Exception:
        return pred == gold


def grade_symbolic(pred, gold):
    """math-verify: parse both as LaTeX (gold inside $...$, the prediction as a boxed answer) and
    test equivalence (sympy simplification, numeric tolerance, set and interval handling)."""
    if pred is None:
        return False
    from math_verify import parse, verify
    try:
        g = parse("$" + gold + "$")
        p = parse("\\boxed{" + pred + "}")
        return bool(verify(g, p))
    except Exception:
        return False


GRADERS = {"exact": grade_exact, "normalized": grade_normalized, "symbolic": grade_symbolic}
