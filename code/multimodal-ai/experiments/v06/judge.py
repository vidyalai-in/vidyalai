"""Claude Haiku 5.5 as a prompt-adherence judge for the guidance sweep (thinking disabled).

  judge.py count               free count_tokens for one image request
  judge.py pilot N             N single calls through ledger.create -> runs/judge_pilot.jsonl
  judge.py batch               one Batch API run over every guidance-sweep image not yet judged -> runs/judge.jsonl
"""
import base64, io, json, os, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
from PIL import Image

PART, MODEL, MAX_TOKENS = "v06", "claude-haiku-5-5", 16
HERE = os.path.dirname(os.path.abspath(__file__))
Q = ('This image was generated from the caption: "{p}". Does the image show everything the caption asks for: '
     'each named object, its color, its number and how the objects are arranged? '
     'Reply with exactly one word: yes or no.')


def b64(path):
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


QN = 'Are there any {c} visible in this image? Reply with exactly one word: yes or no.'


def params(rec):
    sweep = "negative" if "concept" in rec else "guidance"
    text = QN.format(c=rec["concept"]) if sweep == "negative" else Q.format(p=rec["prompt"])
    return dict(model=MODEL, max_tokens=MAX_TOKENS, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": b64(f"{HERE}/runs/img/{sweep}/{rec['key'].removeprefix('neg_')}.png")}},
                    {"type": "text", "text": text}]}])


def parse(text):
    t = text.strip().lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else None


recs = [json.loads(l) for l in open(f"{HERE}/runs/guidance.jsonl")]
for l in open(f"{HERE}/runs/negative.jsonl"):
    r = json.loads(l); r["key"] = "neg_" + r["key"]; recs.append(r)
cl = ledger.client()
cmd = sys.argv[1]
if cmd == "count":
    p = params(recs[0]); p.pop("max_tokens"); p.pop("thinking")
    print(cl.messages.count_tokens(**p).input_tokens)
elif cmd == "pilot":
    out = open(f"{HERE}/runs/judge_pilot.jsonl", "a")
    for r in recs[:int(sys.argv[2])]:
        msg = ledger.create(cl, PART, note=f"v06 judge pilot {r['key']}", **params(r))
        text = "".join(c.text for c in msg.content if c.type == "text")
        row = dict(key=r["key"], text=text, verdict=parse(text), stop=msg.stop_reason, usage=msg.usage.model_dump())
        out.write(json.dumps(row) + "\n"); print(row, flush=True)
    print(ledger.report())
elif cmd == "batch":
    path = f"{HERE}/runs/judge.jsonl"
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    todo = [r for r in recs if r["key"] not in done and not r["nsfw"]]
    reqs = [{"custom_id": r["key"].replace(".", "_"), "params": params(r)} for r in todo]
    ids = {q["custom_id"]: r["key"] for q, r in zip(reqs, todo)}
    est = sum(ledger.estimate_cost(MODEL, 450, MAX_TOKENS) for _ in reqs) * 0.5
    print(len(reqs), "requests, worst case $", round(est, 4))
    ledger.guard(PART, est)
    b = cl.messages.batches.create(requests=reqs)
    open(f"{HERE}/runs/judge_batch_ids.txt", "a").write(b.id + "\n")
    while True:
        time.sleep(20)
        b = cl.messages.batches.retrieve(b.id)
        print(b.processing_status, b.request_counts.model_dump(), flush=True)
        if b.processing_status == "ended":
            break
    out = open(path, "a")
    for item in cl.messages.batches.results(b.id):
        if item.result.type != "succeeded":
            out.write(json.dumps(dict(key=ids[item.custom_id], error=item.result.type)) + "\n"); continue
        msg = item.result.message
        ledger.record(PART, MODEL, msg.usage, note="v06 judge batch", batch=True)
        text = "".join(c.text for c in msg.content if c.type == "text")
        out.write(json.dumps(dict(key=ids[item.custom_id], text=text, verdict=parse(text), stop=msg.stop_reason,
                                  usage=msg.usage.model_dump())) + "\n")
    print(ledger.report())
