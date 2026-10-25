"""Part 16: a docs-maintenance agent, built from the pieces the series measured.

The agent checks one documentation page at a time for three kinds of defect:

  link   a cross-reference (:ref:, :doc:, :meth:, :func:, :class: ...) whose target does not exist
  code   a doctest example whose output no longer matches what the library does
  fact   a sentence that contradicts the library's source code (the source of truth)

and proposes a fix for each as an exact text replacement. It never edits the docs: findings go
to a list a human reviews (and make_patch() turns them into a unified diff).

What each earlier part contributes:

  Part 2   strict tool schemas (harness.Tool), descriptions that say when to use each tool
  Part 3   adaptive thinking with a low effort level
  Part 4   a verifier tool: check_fix applies a proposed fix to a scratch copy and re-runs the checks
  Part 5   small tool results: line ranges, capped search hits, one page per run, prompt caching
  Part 12  reliability.run_durable: checkpoints, step and cost budgets, a repeated-failure rule;
           report_issue is idempotent (one finding per page, line and kind)
  Part 13  checks run in a bubblewrap sandbox with no network and a read-only view of the corpus;
           a permission policy (allow/deny) sits in front of every tool call
  Part 11  repeats and Wilson intervals (evals.py), in the experiment scripts

harness.py, reliability.py and ledger.py are imported, not changed.
"""

import json
import re
import resource
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import tool  # noqa: E402
from reliability import RepeatedFailureRule, run_durable  # noqa: E402

CHECKERS = Path("/home/amit/.cache/vidyalai-scratch/a16/tools")  # linkcheck.py, doccheck.py
MAX_LINES = 150      # read_doc / read_source return at most this many lines per call
MAX_HITS = 40        # search_source returns at most this many matching lines
ADAPTIVE = {"type": "adaptive"}

SYSTEM = """You maintain the documentation of `semver`, a Python library for semantic versioning.
You review ONE documentation page (reStructuredText, Sphinx) per task and report its defects.

Look for exactly three kinds of defect:
- link: a cross-reference (:ref:, :doc:, :meth:, :func:, :class:, :exc: ...) whose target does not exist.
- code: a doctest example (lines starting with >>>) whose expected output no longer matches what the
  library actually does, or that raises an error.
- fact: a statement in the prose that contradicts the library's source code: a wrong default value,
  parameter name, exception type, behaviour or list of supported options.

The source code under src/semver/ (and CHANGELOG.rst) is the source of truth. When the page and the
code disagree, the page is wrong. Do not report spelling, grammar, style, missing content, or claims
you cannot check against the code.

How to work:
1. Read the whole page with read_doc.
2. Run check_links and run_examples on it. Their failures are real; you only need to propose fixes.
3. For every factual claim in the prose (defaults, parameters, exceptions, behaviour, operators),
   find the code that implements it with search_source and read_source, and compare.
4. For each defect, write the smallest fix as an exact replacement: old_text must be copied exactly
   from the page and occur once in it. For link and code fixes, call check_fix first and only
   report a fix that makes the failure go away.
5. Call report_issue once per defect. Then reply with a one-line summary. If the page has no
   defects, report nothing and say so.
"""


# ---- The sandbox (Part 13): checks run with no network and a read-only corpus -------------------

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GB address space
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))


def sandboxed(corpus, argv, overlay=None, timeout=30):
    """Run a checker in bubblewrap: new namespaces (no network), empty environment, read-only
    /usr, corpus and checkers; `overlay` (host file, page path) mounts an edited copy of one page."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONPATH", "/corpus/src",
           "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--ro-bind", str(corpus), "/corpus", "--ro-bind", str(CHECKERS), "/checkers"]
    if overlay:
        cmd += ["--ro-bind", str(overlay[0]), f"/corpus/docs/{overlay[1]}"]
    cmd += ["--chdir", "/corpus", "/usr/bin/python3", *argv]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=_limits,
                       env={"PATH": "/usr/bin:/bin"})
    if p.returncode != 0:
        raise RuntimeError(f"checker failed: {p.stderr.strip()[-400:]}")
    return json.loads(p.stdout)


def link_report(corpus, page, overlay=None):
    return sandboxed(corpus, ["/checkers/linkcheck.py", "docs", f"docs/{page}"], overlay)


def example_report(corpus, page, overlay=None):
    return sandboxed(corpus, ["/checkers/doccheck.py", f"docs/{page}"], overlay)


def fmt_links(r):
    if not r["broken"]:
        return f"All {r['links']} cross-references resolve."
    out = [f"{len(r['broken'])} of {r['links']} cross-references are broken:"]
    for b in r["broken"]:
        close = f"  close matches: {', '.join(b['close_matches'])}" if b["close_matches"] else "  no close match"
        out.append(f"line {b['line']}: :{b['role']}: target '{b['target']}' does not exist.{close}")
    return "\n".join(out)


def fmt_examples(r):
    if not r["failures"]:
        return f"All {r['examples']} doctest examples pass."
    out = [f"{len(r['failures'])} of {r['examples']} doctest examples fail:"]
    for f in r["failures"]:
        out.append(f"line {f['line']}: >>> {f['source']}\n  expected: {f['expected']}\n  got:      {f['got']}")
    return "\n".join(out)


# ---- The permission policy (Part 13) -----------------------------------------------------------

def make_policy(corpus, page):
    """allow or deny for one call {name, input}. The agent has no tool that writes to the docs, so
    nothing needs a human 'ask'; the policy keeps every path inside the corpus and keeps
    check_fix and report_issue on the page under review."""
    docs, src = (corpus / "docs").resolve(), corpus.resolve()

    def inside(path, root):
        p = (root / path).resolve()
        return p == root or root in p.parents

    def policy(call):
        name, a = call["name"], call["input"]
        if name in ("check_fix", "report_issue") and a.get("path") != page:
            return "deny"
        if "path" in a:
            root = src if name == "read_source" else docs
            if not inside(a["path"], root):
                return "deny"
            if name == "read_source" and not (a["path"].startswith("src/") or a["path"] == "CHANGELOG.rst"):
                return "deny"
        return "allow"
    return policy


# ---- The tools (Parts 2 and 5) -----------------------------------------------------------------

def _match_ignoring_indent(text, old_text):
    """Find a multi-line old_text in text when only leading spaces differ. Returns (start, end,
    indents) if exactly one place matches, else None. indents[j] is the page's indentation for
    line j of the replacement (line 0 continues the page's line, so it gets none)."""
    old = old_text.split("\n")
    if len(old) < 2:
        return None
    lines = text.split("\n")
    offs, o = [], 0
    for l in lines:
        offs.append(o)
        o += len(l) + 1
    ind = lambda l: l[: len(l) - len(l.lstrip(" "))]  # noqa: E731
    hits = []
    for i in range(len(lines) - len(old) + 1):
        first, last = old[0].lstrip(" "), old[-1].lstrip(" ")
        if not lines[i].endswith(first):
            continue
        if any(lines[i + j].lstrip(" ") != old[j].lstrip(" ") for j in range(1, len(old) - 1)):
            continue
        tail = lines[i + len(old) - 1]
        if not tail.lstrip(" ").startswith(last):
            continue
        start = offs[i] + len(lines[i]) - len(first)
        end = offs[i + len(old) - 1] + len(ind(tail)) + len(last)
        hits.append((start, end, [""] + [ind(lines[i + j]) for j in range(1, len(old))]))
    return hits[0] if len(hits) == 1 else None


def make_tools(corpus, page, findings, kit="full", decisions=None):
    """Tools for reviewing one page of the corpus at `corpus` (a folder with docs/, src/semver/ and
    CHANGELOG.rst). Reported issues are appended to `findings`. kit 'full' has every tool; 'read'
    drops the three deterministic checks (check_links, run_examples, check_fix). 'full2' is 'full' with
    lenient text matching: a multi-line old_text whose lines differ from the page only in leading
    spaces is accepted, and the page's own indentation is kept."""
    corpus = Path(corpus).resolve()
    docs = corpus / "docs"
    policy = make_policy(corpus, page)
    decisions = decisions if decisions is not None else []
    lock = threading.Lock()
    lenient = kit == "full2"

    def numbered(path, start, end):
        lines = path.read_text().splitlines()
        end = len(lines) if end == -1 else min(end, len(lines))
        start = max(1, start)
        end = min(end, start + MAX_LINES - 1)
        body = "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n(lines {end + 1}-{len(lines)} not shown)" if end < len(lines) else ""
        return f"{path.name}: lines {start}-{end} of {len(lines)}\n{body}{more}"

    @tool("Read lines start_line..end_line (1-based, inclusive) of a documentation page, with line "
          f"numbers. Paths are relative to the docs folder, e.g. 'usage/parse-version-string.rst'. At most "
          f"{MAX_LINES} lines per call; use end_line -1 for the end of the page.",
          path={"type": "string"}, start_line={"type": "integer"}, end_line={"type": "integer"})
    def read_doc(path, start_line, end_line):
        return numbered(docs / path, start_line, end_line)

    @tool("Check every cross-reference on a documentation page (:ref:, :doc:, and Python roles such as "
          ":meth:, :func:, :class:) against the docs folder and the installed library. Returns the broken "
          "ones with their line numbers and the closest existing targets.", path={"type": "string"})
    def check_links(path):
        return fmt_links(link_report(corpus, path))

    @tool("Run the doctest examples (>>> lines) of a documentation page against the library, in a "
          "sandbox. Returns each failing example with its line, expected output and actual output.",
          path={"type": "string"})
    def run_examples(path):
        return fmt_examples(example_report(corpus, path))

    @tool("Search the library source (src/semver/*.py) and CHANGELOG.rst with a Python regular "
          f"expression. Returns up to {MAX_HITS} matching lines as 'file:line: text'. Use it to find the "
          "code behind a claim (a default value, a parameter, an exception), then read_source.",
          pattern={"type": "string"})
    def search_source(pattern):
        rx = re.compile(pattern)
        hits = []
        for f in sorted((corpus / "src").rglob("*.py")) + [corpus / "CHANGELOG.rst"]:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(corpus)}:{i}: {line.strip()[:200]}")
        more = f"\n... {len(hits) - MAX_HITS} more; narrow the pattern." if len(hits) > MAX_HITS else ""
        return "\n".join(hits[:MAX_HITS]) + more if hits else "No matches."

    @tool(f"Read lines start_line..end_line of a source file, e.g. 'src/semver/version.py' or "
          f"'CHANGELOG.rst', with line numbers. At most {MAX_LINES} lines per call; end_line -1 for the end.",
          path={"type": "string"}, start_line={"type": "integer"}, end_line={"type": "integer"})
    def read_source(path, start_line, end_line):
        return numbered(corpus / path, start_line, end_line)

    def apply(path, old_text, new_text):
        text = (docs / path).read_text()
        n = text.count(old_text)
        if n != 1 and lenient:
            hit = _match_ignoring_indent(text, old_text)
            if hit is not None:  # same lines with different leading whitespace: use the page's own
                start, end, indent = hit
                new_lines = new_text.split("\n")
                new_text = "\n".join(indent[min(i, len(indent) - 1)] + l.lstrip(" ")
                                     for i, l in enumerate(new_lines))
                return text[:start] + new_text + text[end:]
        if n != 1:
            raise ValueError(f"old_text occurs {n} times in {path}; it must occur exactly once. "
                             "Copy it exactly from read_doc output (without the line numbers).")
        return text.replace(old_text, new_text)

    @tool("Verify a proposed fix without changing the page: replaces old_text (which must occur exactly "
          "once on the page) with new_text in a scratch copy, re-runs check_links and run_examples on the "
          "copy, and reports what was failing before and after.",
          path={"type": "string"}, old_text={"type": "string"}, new_text={"type": "string"})
    def check_fix(path, old_text, new_text):
        fixed = apply(path, old_text, new_text)
        with tempfile.NamedTemporaryFile("w", suffix=".rst", delete=False) as tmp:
            tmp.write(fixed)
        try:
            before_l, before_e = link_report(corpus, path), example_report(corpus, path)
            after_l = link_report(corpus, path, (tmp.name, path))
            after_e = example_report(corpus, path, (tmp.name, path))
        finally:
            Path(tmp.name).unlink()
        return (f"Before: {len(before_l['broken'])} broken links, {len(before_e['failures'])} failing examples. "
                f"After: {len(after_l['broken'])} broken links, {len(after_e['failures'])} failing examples.\n"
                f"After the fix:\n{fmt_links(after_l)}\n{fmt_examples(after_e)}")

    @tool("Report one defect on the page under review, with a proposed fix. kind is 'link', 'code' or "
          "'fact'. line is the line number where the defect is. old_text is copied exactly from the page "
          "and must occur once; new_text replaces it. Reporting the same page, line and kind again "
          "replaces the earlier report.",
          path={"type": "string"}, line={"type": "integer"},
          kind={"type": "string", "enum": ["link", "code", "fact"]},
          problem={"type": "string", "description": "One sentence: what is wrong and what the code says."},
          old_text={"type": "string"}, new_text={"type": "string"})
    def report_issue(path, line, kind, problem, old_text, new_text):
        fixed = apply(path, old_text, new_text)  # validates old_text
        if lenient and (docs / path).read_text().count(old_text) != 1:  # store the exact page text
            old_text, new_text = _exact_edit((docs / path).read_text(), fixed)
        key = (path, line, kind)
        rec = dict(path=path, line=line, kind=kind, problem=problem, old_text=old_text, new_text=new_text)
        with lock:
            for i, f in enumerate(findings):
                if (f["path"], f["line"], f["kind"]) == key:  # idempotent: same key replaces
                    findings[i] = dict(rec, replaced=f.get("replaced", 0) + 1)
                    return f"Updated issue #{i + 1} ({kind}, line {line})."
            findings.append(rec)
            return f"Recorded issue #{len(findings)} ({kind}, line {line})."

    kits = {"full": [read_doc, check_links, run_examples, search_source, read_source, check_fix, report_issue],
            "read": [read_doc, search_source, read_source, report_issue]}
    tools = kits["full" if kit == "full2" else kit]

    # Every call passes the policy first; a denied call returns an error to the model.
    for t in tools:
        def guarded(_fn=t.fn, _name=t.name, **kw):
            d = policy({"name": _name, "input": kw})
            decisions.append((_name, d))
            if d == "deny":
                raise PermissionError(f"Denied by policy: {_name} on {kw.get('path')!r} is not allowed.")
            return _fn(**kw)
        t.fn = guarded
    return tools


# ---- One page, one run (Part 12's durable loop) ---------------------------------------------------

def review_page(corpus, page, *, part, client, model="claude-haiku-5-5", effort="low",
                thinking=ADAPTIVE, kit="full", max_steps=30, max_cost=0.05, checkpoint=None,
                transcript=None):
    """Review one page. Returns (findings, result, decisions). With `checkpoint`, a crashed run resumes
    from its last step, findings included."""
    findings, decisions = [], []
    if checkpoint and Path(checkpoint).exists():
        saved = json.loads(Path(checkpoint).read_text()).get("state") or []
        findings.extend(saved)
    tools = make_tools(corpus, page, findings, kit, decisions)
    task = f"Review the documentation page '{page}'."
    result = run_durable(task, tools, SYSTEM, part=part, client=client, model=model, effort=effort,
                         thinking=thinking, max_tokens=8000, max_steps=max_steps, max_cost=max_cost,
                         rule=RepeatedFailureRule(3), checkpoint=checkpoint, state=lambda: findings,
                         transcript=transcript, note=f"{part} {page}")
    return findings, result, decisions


def _exact_edit(text, fixed):
    """The smallest (old_text, new_text) pair, in whole lines, that turns text into fixed."""
    a, b = text.split("\n"), fixed.split("\n")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    while text.count("\n".join(a[i:len(a) - j])) != 1 and (i > 0 or j > 0):  # widen until unique
        i, j = max(0, i - 1), max(0, j - 1)
    return "\n".join(a[i:len(a) - j]), "\n".join(b[i:len(b) - j])


def make_patch(corpus, findings):
    """A unified diff of every proposed fix, for a human to review and apply."""
    import difflib
    out = []
    by_page = {}
    for f in findings:
        by_page.setdefault(f["path"], []).append(f)
    for page, fs in sorted(by_page.items()):
        text = (Path(corpus) / "docs" / page).read_text()
        new = text
        for f in fs:
            if new.count(f["old_text"]) == 1:
                new = new.replace(f["old_text"], f["new_text"])
        out += difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/docs/{page}", f"b/docs/{page}")
    return "".join(out)
