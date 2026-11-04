"""Shared API helpers for e08: request builders, single calls (pilot) and batch runs, all via the ledger."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger

PART = "e08"
RUNS = Path("/home/amit/.cache/vidyalai-scratch/e08/runs")
D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
OFF = {"type": "disabled"}           # Haiku 5.5: thinking off
SONNET_LOW = {"type": "between_tools"}  # Sonnet 5.5's lowest setting (rejects "disabled")
ADAPTIVE = {"type": "adaptive"}

CODE_INSTR = ("Complete the following Python function. Reply with the complete function, including its "
              "signature and any imports it needs, in a single ```python code block, and nothing else.\n\n")
MATH_INSTR = "Solve the following math problem. Put your final answer in \\boxed{}.\n\n"


def code_params(task, model=HAIKU, thinking=OFF, max_tokens=1024):
    return dict(model=model, max_tokens=max_tokens, thinking=thinking,
                messages=[{"role": "user", "content": CODE_INSTR + "```python\n" + task["prompt"] + "```"}])


def math_params(item, model=HAIKU, thinking=OFF, effort=None, max_tokens=4096):
    p = dict(model=model, max_tokens=max_tokens, thinking=thinking,
             messages=[{"role": "user", "content": MATH_INSTR + item["problem"]}])
    if effort:
        p["output_config"] = {"effort": effort}
    return p


def text_of(msg):
    return "".join(b.text for b in msg.content if b.type == "text")


def single(params, note, log):
    r = ledger.create(ledger.client(), PART, note=note, **params)
    rec = dict(params=params, stop=r.stop_reason, text=text_of(r), usage=r.usage.model_dump(),
               types=[b.type for b in r.content])
    with open(RUNS / log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_batch(reqs, model, note, out_name, poll=20):
    """reqs: list of (custom_id, params). Writes the batch id first, then every result as JSONL."""
    cl = ledger.client()
    est = sum(ledger.estimate_cost(model, len(json.dumps(p)) // 2, p["max_tokens"]) for _, p in reqs) * 0.5
    ledger.guard(PART, est)
    b = cl.messages.batches.create(requests=[{"custom_id": c, "params": p} for c, p in reqs])
    (RUNS / f"{out_name}.batch_id").write_text(b.id)
    print("batch", b.id, "est", round(est, 4), flush=True)
    return collect(b.id, model, note, out_name, {c: p for c, p in reqs}, poll)


def collect(batch_id, model, note, out_name, params_by_id=None, poll=20):
    cl = ledger.client()
    t0 = time.time()
    while True:
        b = cl.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        print(f"  {int(time.time()-t0)} s {b.request_counts.model_dump()}", flush=True)
        time.sleep(poll)
    out = RUNS / f"{out_name}.jsonl"
    n = 0
    with out.open("w") as f:
        for it in cl.messages.batches.results(batch_id):
            rec = dict(id=it.custom_id, type=it.result.type)
            if it.result.type == "succeeded":
                m = it.result.message
                ledger.record(PART, model, m.usage, note=note, batch=True)
                rec.update(stop=m.stop_reason, text=text_of(m), usage=m.usage.model_dump(),
                           types=[x.type for x in m.content])
                n += 1
            else:
                rec["error"] = it.result.model_dump(exclude_none=True)
            if params_by_id:
                rec["params"] = params_by_id.get(it.custom_id)
            f.write(json.dumps(rec) + "\n")
    print("done", n, "succeeded in", int(time.time() - t0), "s", flush=True)
    return out
