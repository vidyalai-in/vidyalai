"""Part 5: context engineering on top of the Part 1 harness.

One agent loop, six ways to manage what goes into the context window:

  naive     full tool results, full history (the Part 1 loop)
  trunc     the same tools, each result cut to its first 2,000 characters
  sized     tools designed to return small results: key fields, pages and handles
  edit      full results; the API clears old tool results once the prompt passes 30,000 tokens
            (context editing, beta context-management-2025-06-27)
  compact   full results; the API summarizes the history once the prompt passes 50,000 tokens
            (server-side compaction, beta compact-2026-01-12)
  memory    full results plus the memory tool; when the prompt passes 30,000 tokens the harness
            starts a fresh context window and the model reloads its notes from /memories

harness.py, react.py and ledger.py are imported, not changed.
"""

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Transcript, run_tool  # noqa: E402

MODEL = "claude-haiku-5-5"
ADAPTIVE = {"type": "adaptive"}  # thinking display defaults to "omitted" on Haiku 5.5

EDIT = {"edits": [{"type": "clear_tool_uses_20250919",
                   "trigger": {"type": "input_tokens", "value": 30_000},
                   "keep": {"type": "tool_uses", "value": 4},
                   "clear_at_least": {"type": "input_tokens", "value": 10_000}}]}
COMPACT = {"edits": [{"type": "compact_20260112",
                      "trigger": {"type": "input_tokens", "value": 50_000}}]}  # 50,000 is the minimum
BETAS = {"edit": ["context-management-2025-06-27"], "compact": ["compact-2026-01-12"]}
CONTEXT = {"edit": EDIT, "compact": COMPACT}
MEMORY_TOOL = {"type": "memory_20250818", "name": "memory"}
MEMORY_RULES = """
Your context window is reset whenever it fills up, and anything not saved in your memory directory
is lost. As you work, keep a progress file in /memories: which records you have read, the values you
need from each, and what to read next."""
RESUME = "\n\n(A previous context window already worked on this task. Check your memory directory first.)"


# ---- One API call, with the same accounting as ledger.create --------------------------------

def call(cl, part, note="", betas=None, **kw):
    """messages.create (or beta.messages.create with `betas`): budget check first, then one ledger
    entry for the request and one more for each compaction iteration it ran."""
    model = kw["model"]
    api = cl.beta.messages if betas else cl.messages
    keys = ("model", "messages", "system", "tools", "thinking") + (("context_management",) if betas else ())
    count = {k: v for k, v in kw.items() if k in keys}
    if betas:
        count["betas"] = betas
    est_in = api.count_tokens(**count).input_tokens
    r = ledger._rates(model, est_in)
    worst = (est_in * r["cache_write"] + kw["max_tokens"] * r["out"]) / 1e6
    if betas and "compact" in betas[0]:
        worst *= 2  # a compaction iteration reads the whole history once more
    ledger.guard(part, worst)
    t0 = time.perf_counter()
    resp = api.create(betas=betas, **kw) if betas else api.create(**kw)
    latency = time.perf_counter() - t0
    usage = resp.usage.model_dump()
    cost = ledger.record(part, model, usage, note)  # top-level usage: every non-compaction iteration
    for it in usage.get("iterations") or []:
        if it.get("type") == "compaction":
            cost += ledger.record(part, model, it, note + " compaction")
    return resp, usage, cost, latency


# ---- The memory tool's client side ----------------------------------------------------------

class MemoryStore:
    """A minimal handler for the memory tool's commands, confined to one directory."""
    def __init__(self, root):
        self.root = Path(root)
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "memories").mkdir(parents=True)

    def _path(self, p):
        full = (self.root / p.lstrip("/")).resolve()
        if not str(full).startswith(str((self.root / "memories").resolve())):
            raise ValueError(f"Path {p} is outside /memories")
        return full

    def __call__(self, command, path=None, **a):
        if command == "view":
            f = self._path(path)
            if f.is_dir():
                items = [f"{x.stat().st_size / 1024:.1f}K\t/{x.relative_to(self.root)}" for x in sorted(f.rglob("*"))]
                return ("Here're the files and directories up to 2 levels deep in "
                        f"{path}, excluding hidden items and node_modules:\n" + "\n".join([f"4.0K\t{path}"] + items))
            if not f.exists():
                return f"The path {path} does not exist. Please provide a valid path."
            lines = f.read_text().splitlines()
            lo, hi = (a.get("view_range") or [1, -1])
            hi = len(lines) if hi == -1 else hi
            return f"Here's the content of {path} with line numbers:\n" + "\n".join(
                f"{i:6d}\t{lines[i - 1]}" for i in range(lo, hi + 1) if 0 < i <= len(lines))
        if command == "create":
            f = self._path(path)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(a["file_text"])
            return f"File created successfully at: {path}"
        if command == "str_replace":
            f = self._path(path)
            if not f.is_file():
                return f"Error: The path {path} does not exist. Please provide a valid path."
            text, old = f.read_text(), a["old_str"]
            if text.count(old) != 1:
                return (f"No replacement was performed, old_str `{old}` did not appear verbatim in {path}."
                        if old not in text else f"No replacement was performed. Multiple occurrences of old_str `{old}`.")
            f.write_text(text.replace(old, a.get("new_str", "")))
            return "The memory file has been edited."
        if command == "insert":
            f = self._path(path)
            if not f.is_file():
                return f"Error: The path {path} does not exist"
            lines = f.read_text().splitlines()
            n = a["insert_line"]
            lines[n:n] = a["insert_text"].rstrip("\n").splitlines()
            f.write_text("\n".join(lines) + "\n")
            return f"The file {path} has been edited."
        if command == "delete":
            f = self._path(path)
            if f == self._path("/memories"):
                return "Error: cannot delete the /memories directory"
            if not f.exists():
                return f"Error: The path {path} does not exist"
            shutil.rmtree(f) if f.is_dir() else f.unlink()
            return f"Successfully deleted {path}"
        if command == "rename":
            src, dst = self._path(a["old_path"]), self._path(a["new_path"])
            if not src.exists():
                return f"Error: The path {a['old_path']} does not exist"
            if dst.exists():
                return f"Error: The destination {a['new_path']} already exists"
            src.rename(dst)
            return f"Successfully renamed {a['old_path']} to {a['new_path']}"
        return f"Error: unknown command {command}"


# ---- The loop ---------------------------------------------------------------------------------

@dataclass
class CStep:
    n: int
    window: int            # context window number (the memory arm starts new ones)
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    context: int           # tokens the model read this step, after any clearing or compaction
    cleared: int           # tokens cleared by context editing this step
    compacted: bool        # a compaction iteration ran in this request
    compaction_in: int     # tokens that iteration read
    latency_s: float
    cost: float
    stop_reason: str
    tool_calls: list


@dataclass
class CResult:
    answer: str
    stop: str
    steps: list = field(default_factory=list)

    @property
    def cost(self):
        return sum(s.cost for s in self.steps)


def run(task, tools, system, arm, *, part, client, model=MODEL, effort="medium", window=30_000,
        max_steps=90, max_cost=0.25, max_tokens=8000, memdir=None, transcript=None,
        cache=True, clock=False):
    """Run one task with one context strategy. `tools` is a list of harness Tools.
    cache=False sends no cache_control at all; clock=True puts the current time at the top of the
    system prompt on every request (a classic way to break the cache)."""
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    store = None
    if arm == "memory":
        store = MemoryStore(memdir)
        schemas.append(MEMORY_TOOL)
        system = system + MEMORY_RULES
    request = dict(model=model, max_tokens=max_tokens, tools=schemas, thinking=ADAPTIVE,
                   system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, cache_control={"type": "ephemeral"})
    if not cache:
        del request["cache_control"], request["system"][0]["cache_control"]
    betas = BETAS.get(arm)
    if arm in CONTEXT:
        request["context_management"] = CONTEXT[arm]
    log = Transcript(transcript)
    log.write(event="task", arm=arm, task=task, betas=betas, **request)

    messages, win = [{"role": "user", "content": task}], 1
    result = CResult(answer="", stop="max_steps")
    for n in range(1, max_steps + 1):
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        if clock:
            request["system"][0]["text"] = f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{system}"
        resp, u, cost, latency = call(client, part, note=f"a05 {arm} step {n}", betas=betas,
                                      messages=messages, **request)
        calls = [b for b in resp.content if b.type == "tool_use"]
        cm = getattr(resp, "context_management", None)
        cleared = sum((e.cleared_input_tokens or 0) for e in (cm.applied_edits if cm else []) or [])
        comp = [it for it in (u.get("iterations") or []) if it.get("type") == "compaction"]
        ctx = u["input_tokens"] + (u["cache_read_input_tokens"] or 0) + (u["cache_creation_input_tokens"] or 0)
        step = CStep(n, win, u["input_tokens"], u["cache_read_input_tokens"] or 0,
                     u["cache_creation_input_tokens"] or 0, u["output_tokens"], ctx, cleared, bool(comp),
                     sum(it.get("input_tokens", 0) + (it.get("cache_read_input_tokens") or 0)
                         + (it.get("cache_creation_input_tokens") or 0) for it in comp),
                     round(latency, 3), cost, resp.stop_reason,
                     [f"{b.name}({json.dumps(b.input)})" for b in calls])
        result.steps.append(step)
        log.write(event="step", **step.__dict__, usage=u,
                  content=[b.model_dump(exclude_none=True) for b in resp.content])

        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = []
        for b in calls:
            if b.name == "memory" and store:
                try:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": store(**b.input)})
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": f"Error: {e}", "is_error": True})
            else:
                results.append(run_tool(by_name, b))
        messages.append({"role": "user", "content": results})
        log.write(event="tool_results", n=n, results=results)
        if store and ctx > window:  # the memory arm: start a fresh context window
            win += 1
            messages = [{"role": "user", "content": task + RESUME}]
            log.write(event="reset", n=n, window=win, memory=sorted(
                str(p.relative_to(store.root)) for p in store.root.rglob("*") if p.is_file()))

    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps),
              cost=result.cost)
    return result
