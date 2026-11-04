"""Claude Haiku 5.5 on the synthetic task set.

  claude_run.py count            free count_tokens for each image size (+ a text-only baseline)
  claude_run.py pilot N          N single calls through ledger.create
  claude_run.py batch OUT SIZES  one Batch API run over all items at the given sizes (resumable by id)
"""
import json, os, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import vlm, ledger

PART = "v04"
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")
# THINK: "disabled" (default), "default" (omit the field: Haiku 5.5 then thinks), or another type
# such as "between_tools" (Sonnet 5.5 rejects "disabled").
THINK = os.environ.get("THINK", "disabled")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
# TERSE=1 adds an instruction not to count aloud (tests whether listing positions is what makes counting work)
TERSE = " Do not list, count aloud or explain: reply with the number immediately." if os.environ.get("TERSE") else ""
HERE = os.path.dirname(os.path.abspath(__file__))


def params(it, size):
    img = vlm.render(it, size)
    extra = {} if THINK == "default" else {"thinking": {"type": THINK}}
    return dict(model=MODEL, max_tokens=MAX_TOKENS, **extra,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": vlm.png_b64(img)}},
                    {"type": "text", "text": it.question + " " + vlm.SUFFIX[it.task] + TERSE}]}])


def text_only(it):
    return dict(model=MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": it.question + " " + vlm.SUFFIX[it.task]}]}])


cl = ledger.client()
items = vlm.make_items(150)
cmd = sys.argv[1] if len(sys.argv) > 1 else "none"

if cmd == "count":
    rows = []
    for it in [i for i in items if i.id.endswith("-000")]:
        base = cl.messages.count_tokens(**text_only(it)).input_tokens
        for size in (112, 224, 300, 448, 672, 896, 1092, 1344, 1792, 2576, 3000):
            p = params(it, size)
            p.pop("max_tokens"); p.pop("thinking", None)
            n = cl.messages.count_tokens(**p).input_tokens
            w, h = vlm.claude_resized(size, size)
            rows.append(dict(task=it.task, size=size, with_image=n, text_only=base, image=n - base,
                             resized=[w, h], formula=vlm.claude_tokens(w, h)))
            print(rows[-1], flush=True)
    json.dump(rows, open(f"{HERE}/claude_count_tokens.json", "w"), indent=1)

elif cmd == "pilot":
    n = int(sys.argv[2])
    out = open(f"{HERE}/claude_pilot.jsonl", "a")
    picks = [i for i in items if int(i.id[-3:]) < 2][:n]
    for it in picks:
        for size in (224, 896):
            t0 = time.time()
            msg = ledger.create(cl, PART, note=f"v04 pilot {it.id} {size}", **params(it, size))
            text = "".join(c.text for c in msg.content if c.type == "text")
            r = dict(id=it.id, size=size, answer=it.answer, text=text, stop=msg.stop_reason,
                     usage=msg.usage.model_dump(), s=round(time.time() - t0, 2), **vlm.score(it, text))
            out.write(json.dumps(r) + "\n")
            print(r, flush=True)

elif cmd == "batch":
    out_path, sizes = sys.argv[2], [int(s) for s in sys.argv[3:]]
    limit = int(os.environ.get("LIMIT", "150"))
    done = set()
    if os.path.exists(out_path):
        done = {(json.loads(l)["id"], json.loads(l)["size"]) for l in open(out_path)}
    byid = {i.id: i for i in items}
    reqs, est = [], 0.0
    for size in sizes:
        img_tokens = vlm.claude_tokens(*vlm.claude_resized(size, size))
        tasks = os.environ.get("TASKS", ",".join(vlm.TASKS)).split(",")
        for it in items:
            if int(it.id[-3:]) >= limit or (it.id, size) in done or it.task not in tasks:
                continue
            reqs.append({"custom_id": f"{it.id}__{size}", "params": params(it, size)})
            est += ledger.estimate_cost(MODEL, img_tokens + 60, MAX_TOKENS) * 0.5
    print(len(reqs), "requests, worst-case estimate $%.4f" % est, flush=True)
    if not reqs:
        sys.exit(0)
    ledger.guard(PART, est)
    b = cl.messages.batches.create(requests=reqs)
    open(f"{HERE}/batch_ids.txt", "a").write(f"{b.id} {len(reqs)} {sizes} {time.time()}\n")
    t0 = time.time()
    print("batch", b.id, flush=True)
    while True:
        time.sleep(20)
        b = cl.messages.batches.retrieve(b.id)
        if b.processing_status == "ended":
            break
    print("ended after", round(time.time() - t0), "s", b.request_counts, flush=True)
    with open(out_path, "a") as f:
        for item in cl.messages.batches.results(b.id):
            iid, size = item.custom_id.split("__")
            size = int(size)
            if item.result.type != "succeeded":
                f.write(json.dumps(dict(id=iid, size=size, error=item.result.type)) + "\n")
                continue
            msg = item.result.message
            ledger.record(PART, MODEL, msg.usage, note=f"v04 batch {MODEL} {THINK} {size}", batch=True)
            text = "".join(c.text for c in msg.content if c.type == "text")
            n_think = sum(len(getattr(c, "thinking", "") or "") for c in msg.content if c.type == "thinking")
            it = byid[iid]
            f.write(json.dumps(dict(id=iid, task=it.task, size=size, model=MODEL, answer=it.answer, text=text,
                                    stop=msg.stop_reason, usage=msg.usage.model_dump(), think=THINK, think_chars=n_think, terse=bool(TERSE),
                                    turnaround_s=round(time.time() - t0), **vlm.score(it, text))) + "\n")
