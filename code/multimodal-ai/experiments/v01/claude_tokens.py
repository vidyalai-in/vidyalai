"""(a) Claude image tokens: the same synthetic picture at many sizes, counted with the free
count_tokens endpoint (image request minus text-only request), compared with the documented rule.
Then a few real Haiku 5.5 calls (thinking disabled) to check usage.input_tokens.
Usage: python claude_tokens.py count | call
"""
import base64, io, json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import tokens, ledger

MODEL = "claude-haiku-5-5"
PROMPT = "How many red dots are in the row along the bottom of this picture? Reply with the number only."
SIZES = [(150, 100), (300, 200), (600, 400), (900, 600), (1200, 800), (1800, 1200), (2400, 1600), (3000, 2000),
         (4000, 500), (500, 4000)]
OUT = "/home/amit/.cache/vidyalai-scratch/v01/runs"


def b64png(w, h):
    buf = io.BytesIO()
    tokens.scene(w, h).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), len(buf.getvalue())


def msg(img=None):
    content = [{"type": "text", "text": PROMPT}]
    if img:
        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}})
    return [{"role": "user", "content": content}]


cl = ledger.client()
if sys.argv[1] == "count":
    base = cl.messages.count_tokens(model=MODEL, messages=msg()).input_tokens
    rows = []
    for w, h in SIZES:
        data, nbytes = b64png(w, h)
        n = cl.messages.count_tokens(model=MODEL, messages=msg(data)).input_tokens
        rw, rh = tokens.claude_resized_size(w, h)
        r = {"w": w, "h": h, "png_bytes": nbytes, "text_only": base, "with_image": n, "image_tokens": n - base,
             "resized": [rw, rh], "formula": tokens.claude_expected(w, h), "naive_formula": tokens.claude_image_tokens(w, h)}
        rows.append(r)
        print(r, flush=True)
    json.dump(rows, open(f"{OUT}/claude_count.json", "w"), indent=1)
elif sys.argv[1] == "call":
    counts = {(r["w"], r["h"]): r for r in json.load(open(f"{OUT}/claude_count.json"))}
    rows = []
    for w, h in [(300, 200), (1200, 800), (3000, 2000)]:
        data, _ = b64png(w, h)
        t0 = time.time()
        resp = ledger.create(cl, "v01", note=f"image token check {w}x{h}", model=MODEL, max_tokens=16,
                             thinking={"type": "disabled"}, messages=msg(data))
        u = resp.usage.model_dump()
        r = {"w": w, "h": h, "usage": u, "counted": counts[(w, h)]["with_image"], "stop_reason": resp.stop_reason,
             "text": "".join(b.text for b in resp.content if b.type == "text"), "latency_s": time.time() - t0,
             "cost": ledger.cost_of(MODEL, u)}
        rows.append(r)
        print(r, flush=True)
    with open(f"{OUT}/claude_calls.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
