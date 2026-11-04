"""What survives the patch grid: Haiku 5.5 counts the red dots in the synthetic scene at five sizes.
40 scenes (dot count 6-17, seeded), each sent at every size, so sizes are compared on the same items.
Usage: python claude_dots.py pilot | submit | collect
"""
import base64, io, json, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import tokens, ledger

MODEL = "claude-haiku-5-5"
PROMPT = "How many red dots are in the row along the bottom of this picture? Reply with the number only."
SIZES = [(150, 100), (300, 200), (600, 400), (1200, 800), (2400, 1600)]
N_ITEMS = 40
OUT = Path("/home/amit/.cache/vidyalai-scratch/v01/runs")
COUNTS = np.random.default_rng(1).integers(6, 18, size=N_ITEMS).tolist()


def params(w, h, n):
    buf = io.BytesIO()
    tokens.scene(w, h, dots=n).save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"model": MODEL, "max_tokens": 16, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": PROMPT}]}]}


def parse(text):
    m = re.fullmatch(r"\s*(\d+)\s*\.?\s*", text)
    return int(m.group(1)) if m else None


cl = ledger.client()
mode = sys.argv[1]
if mode == "pilot":
    with open(OUT / "dots_pilot.jsonl", "a") as f:
        for i, (w, h) in enumerate(SIZES):
            n = COUNTS[i]
            r = ledger.create(cl, "v01", note=f"dots pilot {w}x{h}", **params(w, h, n))
            text = "".join(b.text for b in r.content if b.type == "text")
            row = {"w": w, "h": h, "n": n, "text": text, "parsed": parse(text), "usage": r.usage.model_dump(),
                   "cost": ledger.cost_of(MODEL, r.usage), "stop": r.stop_reason}
            f.write(json.dumps(row) + "\n")
            print(row["w"], row["n"], repr(text), row["usage"]["input_tokens"], row["cost"], flush=True)
elif mode == "submit":
    reqs = []
    for i, n in enumerate(COUNTS):
        for w, h in SIZES:
            reqs.append({"custom_id": f"i{i:02d}-{w}x{h}-n{n}", "params": params(w, h, n)})
    est = 0.0
    for (w, h) in SIZES:
        nt = tokens.claude_expected(w, h) + 40
        est += N_ITEMS * ledger.estimate_cost(MODEL, nt, 16) * 0.5
    print("estimate", est)
    ledger.guard("v01", est)
    b = cl.messages.batches.create(requests=reqs)
    (OUT / "dots_batch_id.txt").write_text(b.id)
    print("batch", b.id, len(reqs))
elif mode == "collect":
    bid = (OUT / "dots_batch_id.txt").read_text().strip()
    while True:
        b = cl.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        time.sleep(20)
    rows = []
    for item in cl.messages.batches.results(bid):
        i, size, n = item.custom_id.split("-")
        row = {"id": item.custom_id, "item": int(i[1:]), "size": size, "n": int(n[1:]), "type": item.result.type}
        if item.result.type == "succeeded":
            msg = item.result.message
            ledger.record("v01", MODEL, msg.usage, note="dots batch", batch=True)
            text = "".join(c.text for c in msg.content if c.type == "text")
            row.update(text=text, parsed=parse(text), stop=msg.stop_reason, input_tokens=msg.usage.input_tokens,
                       output_tokens=msg.usage.output_tokens, cost=ledger.cost_of(MODEL, msg.usage, batch=True))
        rows.append(row)
    with open(OUT / "dots_batch.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(b.request_counts, len(rows))
