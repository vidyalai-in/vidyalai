"""Part 3 runs: MMLU-1140 multiple choice on Claude, single calls (pilot) or Message Batches.

usage:
  run.py pilot  <config> <n>            n single calls through ledger.create
  run.py batch  <config> <k0> <k1> [limit]   samples k0..k1-1 of every item (or the first `limit`) as one batch
Configs: haiku-off, haiku-low (adaptive thinking, effort low), sonnet-off.
Every request, response and usage line goes to runs/<config>.jsonl.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
try:
    import ledger  # noqa: E402
except ImportError:  # the GPU venv (qwen.py) only needs the prompt
    ledger = None

PART = "e03"
ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
ITEMS = json.loads(Path("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json").read_text())
LETTERS = "ABCD"
SYSTEM = "You are taking a multiple-choice exam. Your entire reply must be one letter: A, B, C or D."

CONFIGS = {
    "haiku-off": dict(model="claude-haiku-5-5", max_tokens=16, thinking={"type": "disabled"}),
    "haiku-low": dict(model="claude-haiku-5-5", max_tokens=2000,
                      thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    "sonnet-off": dict(model="claude-sonnet-5-5", max_tokens=16, thinking={"type": "between_tools"}),
}


def prompt(item):
    subject = item["subject"].replace("_", " ")
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, item["choices"]))
    return (f"The following is a multiple choice question about {subject}.\n\n"
            f"{item['question']}\n{opts}\n\n"
            "Reply with only the letter of the correct option (A, B, C or D). Do not explain.")


def params(config, item):
    return {**CONFIGS[config], "system": SYSTEM, "messages": [{"role": "user", "content": prompt(item)}]}


def extract(text):
    """First standalone A-D in the reply, or None."""
    m = re.search(r"\b([ABCD])\b", text or "")
    return m.group(1) if m else None


def text_of(content):
    return "".join(b["text"] for b in content if b.get("type") == "text")


def log(config, rec):
    with (RUNS / f"{config}.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def row(config, item, k, msg, mode):
    content = [b.model_dump(exclude_none=True) for b in msg.content]
    txt = text_of(content)
    pred = extract(txt)
    return dict(config=config, id=item["id"], subject=item["subject"], k=k, mode=mode,
                text=txt, pred=pred, answer=LETTERS[item["answer"]],
                correct=int(pred == LETTERS[item["answer"]]), stop=msg.stop_reason,
                thinking_blocks=sum(b.get("type") == "thinking" for b in content),
                usage=msg.usage.model_dump(), cost=ledger.cost_of(msg.model, msg.usage, batch=(mode == "batch")))


def pilot(config, n):
    cl = ledger.client()
    for item in ITEMS[::max(1, len(ITEMS) // n)][:n]:
        msg = ledger.create(cl, PART, note=f"e03 pilot {config}", **params(config, item))
        r = row(config, item, -1, msg, "pilot")
        log(config + "-pilot", r)
        print(r["id"], repr(r["text"][:60]), r["pred"], r["answer"], r["usage"]["input_tokens"],
              r["usage"]["output_tokens"], f"${r['cost']:.6f}", r["stop"], r["thinking_blocks"])


def batch(config, k0, k1, limit=None):
    cl = ledger.client()
    items = ITEMS[:limit] if limit else ITEMS
    byid = {it["id"]: it for it in items}
    reqs = [{"custom_id": f"{it['id']}__{k}", "params": params(config, it)} for k in range(k0, k1) for it in items]
    model, mt = CONFIGS[config]["model"], CONFIGS[config]["max_tokens"]
    est = sum(ledger.estimate_cost(model, len(json.dumps(r["params"])) // 2, mt) for r in reqs) * 0.5
    print(f"{len(reqs)} requests, worst-case ${est:.4f}", flush=True)
    ledger.guard(PART, est)
    b = cl.messages.batches.create(requests=reqs)
    (RUNS / f"{config}.batch-{b.id}.json").write_text(json.dumps(dict(id=b.id, n=len(reqs), k0=k0, k1=k1, t=time.time())))
    print("batch", b.id, flush=True)
    t0 = time.time()
    while True:
        time.sleep(20)
        b = cl.messages.batches.retrieve(b.id)
        print(f"{time.time() - t0:6.0f}s {b.processing_status} {b.request_counts.model_dump()}", flush=True)
        if b.processing_status == "ended":
            break
        if time.time() - t0 > 3600:
            print("still running after 60 min", flush=True)
            return
    total, bad = 0.0, 0
    for res in cl.messages.batches.results(b.id):
        iid, k = res.custom_id.rsplit("__", 1)
        if res.result.type != "succeeded":
            bad += 1
            log(config, dict(config=config, id=iid, k=int(k), mode="batch", error=res.result.type))
            continue
        msg = res.result.message
        total += ledger.record(PART, model, msg.usage, note=f"e03 batch {config}", batch=True)
        log(config, row(config, byid[iid], int(k), msg, "batch"))
    print(f"done: cost ${total:.4f}, non-succeeded {bad}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        pilot(sys.argv[2], int(sys.argv[3]))
    elif cmd == "batch":
        batch(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else None)
