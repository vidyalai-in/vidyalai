"""Exp 5 (optional API): does each erroneous transcript change the meaning? Claude Haiku 5.5 judge,
thinking disabled, max_tokens 5, validated against hand labels (runs/hand_labels.json).
Usage: exp5_judge.py items | pilot | batch | collect"""
import sys, json, os, time, random
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import asr
MODEL = "claude-haiku-5-5"
SYSTEM = ("You compare a reference transcript of read English speech with a speech recognizer's transcript. "
          "Both are lowercased with punctuation removed. Decide whether the differences would make a reader "
          "understand something different from what was said: a different word with a different meaning, a missing "
          "or added content word, a wrong name. Spelling variants, a split or joined compound, or a different "
          "inflection that keeps the meaning are harmless. Answer with exactly one word: CHANGED or SAME.")
def prompt(it): return f"Reference: {it['ref']}\nTranscript: {it['hyp']}"
def items():
    out = []
    for m in [m for m in ("tiny", "base", "small", "turbo") if os.path.exists(f"runs/exp1_{m}_b1.jsonl")]:
        for r in map(json.loads, open(f"runs/exp1_{m}_b1.jsonl")):
            ref, hyp = asr.normalize(r["ref"], "english"), asr.normalize(r["hyp"], "english")
            if ref != hyp:
                c = asr.word_errors([ref], [hyp])[0]
                out.append({"cid": f"{m}__{r['id']}", "model": m, "split": r["split"], "ref": ref, "hyp": hyp,
                            "errors": int(c[:3].sum()), "n": int(c[3])})
    return out
cmd = sys.argv[1]
if cmd == "items":
    it = items(); json.dump(it, open("runs/judge_items.json", "w"))
    print(len(it), {m: sum(1 for x in it if x["model"] == m) for m in ("tiny", "base", "small", "turbo")})
    random.seed(0); s = [x for m in ("tiny", "base", "small", "turbo") for x in random.sample([y for y in it if y["model"] == m], 15)]
    json.dump([{"cid": x["cid"], "ref": x["ref"], "hyp": x["hyp"]} for x in s], open("runs/hand_sample.json", "w"), indent=0)
    sys.exit()
import ledger
cl = ledger.client()
it = json.load(open("runs/judge_items.json"))
req = lambda x: dict(model=MODEL, max_tokens=5, system=SYSTEM, thinking={"type": "disabled"},
                     messages=[{"role": "user", "content": prompt(x)}])
if cmd == "pilot":
    with open("runs/judge_pilot.jsonl", "w") as f:
        for x in it[:8]:
            m = ledger.create(cl, "v09", note="v09 judge pilot", **req(x))
            t = "".join(c.text for c in m.content if c.type == "text")
            f.write(json.dumps({"cid": x["cid"], "out": t, "stop": m.stop_reason, "usage": m.usage.model_dump()}) + "\n")
            print(x["cid"], repr(t), m.stop_reason, m.usage.input_tokens, m.usage.output_tokens)
elif cmd == "batch":
    reqs = [{"custom_id": x["cid"].replace(".", "_"), "params": req(x)} for x in it]
    est = sum(ledger.estimate_cost(MODEL, len(prompt(x) + SYSTEM) // 2 + 20, 5) for x in it) * 0.5
    print("estimate $", round(est, 4))
    ledger.guard("v09", est)
    b = cl.messages.batches.create(requests=reqs)
    open("runs/judge_batch_id.txt", "w").write(b.id)
    print("batch", b.id, len(reqs))
elif cmd == "collect":
    bid = open("runs/judge_batch_id.txt").read().strip()
    while True:
        b = cl.messages.batches.retrieve(bid)
        if b.processing_status == "ended": break
        time.sleep(20)
    with open("runs/judge_out.jsonl", "w") as f:
        for item in cl.messages.batches.results(bid):
            if item.result.type != "succeeded":
                f.write(json.dumps({"cid": item.custom_id, "out": None, "type": item.result.type}) + "\n"); continue
            msg = item.result.message
            ledger.record("v09", MODEL, msg.usage, note="v09 meaning judge batch", batch=True)
            t = "".join(c.text for c in msg.content if c.type == "text")
            f.write(json.dumps({"cid": item.custom_id, "out": t, "stop": msg.stop_reason}) + "\n")
    print(b.request_counts)
