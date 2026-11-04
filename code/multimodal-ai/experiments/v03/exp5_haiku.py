"""Exp 5: Claude Haiku 5.5 (thinking disabled) zero-shot on the same images as Exp 4
(rendered shapes; CIFAR-10 subset upscaled to 224 px), clean and with pasted labels.
Usage: python exp5_haiku.py pilot | batch | collect"""
import base64, json, pickle, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ledger

OUT = "/home/amit/.cache/vidyalai-scratch/v03"
MODEL = "claude-haiku-5-5"
data = pickle.load(open(f"{OUT}/typo.pkl", "rb"))


def params(img_bytes, classes):
    q = ("Which one of these labels best describes the image? Labels: " + ", ".join(classes) +
         ". Answer with the label only.")
    return {"model": MODEL, "max_tokens": 10, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(img_bytes).decode()}},
                {"type": "text", "text": q}]}]}


def items():
    for ds, D in data.items():
        for c, ims in D["conds"].items():
            if c == "wrong16": continue
            for i, b in enumerate(ims):
                yield f"{ds}-{c}-{i}", params(b, D["classes"])


cl = ledger.client()
mode = sys.argv[1]
if mode == "pilot":
    for cid, p in list(items())[::700][:6]:
        r = ledger.create(cl, "v03", note="v03 haiku pilot", **p)
        print(cid, repr(r.content[0].text), r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)
    print(ledger.spent("v03"))
elif mode == "batch":
    reqs = [{"custom_id": cid, "params": p} for cid, p in items()]
    n_in = cl.messages.count_tokens(**{k: reqs[0]["params"][k] for k in ("model", "messages", "thinking")}).input_tokens
    est = sum(ledger.estimate_cost(MODEL, n_in + 40, 10) for _ in reqs) * 0.5
    print(len(reqs), "requests; est $", round(est, 4), "input tokens/request ~", n_in)
    ledger.guard("v03", est)
    b = cl.messages.batches.create(requests=reqs)
    open(f"{OUT}/haiku-batch-id.txt", "w").write(b.id)
    print("batch", b.id)
elif mode == "collect":
    bid = open(f"{OUT}/haiku-batch-id.txt").read().strip()
    while cl.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(20)
    n = 0
    with open(f"{OUT}/haiku-results.jsonl", "w") as f:
        for it in cl.messages.batches.results(bid):
            if it.result.type != "succeeded":
                f.write(json.dumps({"id": it.custom_id, "status": it.result.type}) + "\n"); continue
            m = it.result.message
            ledger.record("v03", MODEL, m.usage, note="v03 haiku zero-shot batch", batch=True); n += 1
            f.write(json.dumps({"id": it.custom_id, "status": "ok", "text": "".join(c.text for c in m.content if c.type == "text"),
                                "stop": m.stop_reason, "in": m.usage.input_tokens, "out": m.usage.output_tokens}) + "\n")
    print("recorded", n, "spent v03 $", ledger.spent("v03"))
