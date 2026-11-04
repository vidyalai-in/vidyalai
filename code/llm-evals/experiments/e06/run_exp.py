"""Submit one Part 6 experiment as a Message Batch. Usage: run_exp.py <name>

  rubric1, rubric2  SummEval, anchored rubric, Haiku 5.5, repeats 1 and 2 (1,600 each)
  bare              SummEval, dimension names only, Haiku 5.5 (1,600)
  pair              MT-Bench turn-1 pairs, pair-v2 prompt, both orders (910 x 2)
  verbose           300 sampled pairs: self-duel X vs X+padding, pad answer 1, pad answer 2 (x 2 orders)
  ref               math/reasoning/coding pairs with the GPT-4 reference answer (pair-math-v1), both orders
  single            every turn-1 answer rated 1-10 alone (single-v1)
  sonnet            80 sampled pairs, pair-v2, both orders, Sonnet 5.5

Thinking off for all ({"type": "disabled"} on Haiku, {"type": "between_tools"} on Sonnet).
"""
import json
import random
import sys

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G

D = "/home/amit/.cache/vidyalai-scratch/e06/"
name = sys.argv[1]
se = json.load(open(D + "data/summeval.json"))
mt = json.load(open(D + "data/mtbench.json"))


def sample_pairs(k, seed):
    ok = [i for i, p in enumerate(mt) if len(p["answer_1"]) >= 50 and len(p["answer_2"]) >= 50]
    return sorted(random.Random(seed).sample(ok, k))


jobs = {}
if name in ("rubric1", "rubric2", "bare"):
    for di, d in enumerate(se):
        for si, s in enumerate(d["summaries"]):
            jobs[f"d{di}-s{si}"] = G.rubric_request(d["article"], s, anchored=(name != "bare"))
elif name == "pair":
    for i, p in enumerate(mt):
        jobs[f"p{i}-ab"] = G.pair_request(p["question"], p["answer_1"], p["answer_2"])
        jobs[f"p{i}-ba"] = G.pair_request(p["question"], p["answer_2"], p["answer_1"])
elif name == "verbose":
    for i in sample_pairs(300, seed=6):
        p = mt[i]
        a1, a2, pad = p["answer_1"], p["answer_2"], G.PADDING
        q = p["question"]
        jobs[f"p{i}-self-ab"] = G.pair_request(q, a1, a1 + pad)      # padded copy is B
        jobs[f"p{i}-self-ba"] = G.pair_request(q, a1 + pad, a1)      # padded copy is A
        jobs[f"p{i}-pad1-ab"] = G.pair_request(q, a1 + pad, a2)
        jobs[f"p{i}-pad1-ba"] = G.pair_request(q, a2, a1 + pad)
        jobs[f"p{i}-pad2-ab"] = G.pair_request(q, a1, a2 + pad)
        jobs[f"p{i}-pad2-ba"] = G.pair_request(q, a2 + pad, a1)
elif name == "ref":
    for i, p in enumerate(mt):
        if p["reference"]:
            jobs[f"p{i}-ab"] = G.pair_request(p["question"], p["answer_1"], p["answer_2"], reference=p["reference"])
            jobs[f"p{i}-ba"] = G.pair_request(p["question"], p["answer_2"], p["answer_1"], reference=p["reference"])
elif name == "single":
    seen = {}
    for p in mt:
        for m, a in ((p["model_1"], p["answer_1"]), (p["model_2"], p["answer_2"])):
            seen[(p["qid"], m)] = (p["question"], a)
    models = sorted({m for _, m in seen})
    for (qid, m), (q, a) in sorted(seen.items()):
        jobs[f"q{qid}-m{models.index(m)}"] = G.single_request(q, a)
elif name == "sonnet":
    for i in sample_pairs(80, seed=7):
        p = mt[i]
        jobs[f"p{i}-ab"] = G.pair_request(p["question"], p["answer_1"], p["answer_2"], model="claude-sonnet-5-5", max_tokens=700)
        jobs[f"p{i}-ba"] = G.pair_request(p["question"], p["answer_2"], p["answer_1"], model="claude-sonnet-5-5", max_tokens=700)
elif name == "restate":
    # harder verbosity test: the answer followed by a verbatim restatement of itself
    for i in sample_pairs(300, seed=6)[:150]:
        p = mt[i]
        a1, q = p["answer_1"], p["question"]
        long = a1 + "\n\nTo restate the main points:\n\n" + a1
        jobs[f"p{i}-self-ab"] = G.pair_request(q, a1, long)
        jobs[f"p{i}-self-ba"] = G.pair_request(q, long, a1)
elif name in ("fluency_truecase", "fluency_nocaps"):
    # 200 summaries: re-rated after restoring capitals and spacing (truecase), or with the
    # "no missing capitals" phrase removed from the fluency anchor (nocaps)
    rng = random.Random(66)
    picks = sorted(rng.sample([(di, si) for di in range(100) for si in range(16)], 200))
    for di, si in picks:
        s = se[di]["summaries"][si]
        if name == "fluency_truecase":
            s = G.truecase(s)
            req = G.rubric_request(se[di]["article"], s)
        else:
            req = G.rubric_request(se[di]["article"], s)
            req["system"] = req["system"].replace(", no formatting debris, no missing capitals", ", no formatting debris")
            assert "capitals" not in req["system"]
        jobs[f"d{di}-s{si}"] = req
elif name == "retry":
    # re-ask every reply cut off at max_tokens with a 2,000-token budget (same prompt otherwise)
    run = sys.argv[2]
    for cid, r in G.load_runs(D + f"runs/{run}.jsonl").items():
        if r.get("stop") == "max_tokens":
            jobs[cid] = {**r["params"], "max_tokens": 2000}
    name = f"{run}.retry"
else:
    raise SystemExit(f"unknown experiment {name}")

print(name, len(jobs), "requests", flush=True)
if not jobs:
    raise SystemExit("nothing to send")
res = G.run_batch("e06", jobs, D + f"runs/{name}.jsonl", note=f"e06 {name}")
print(json.dumps(res))
json.dump(res, open(D + f"runs/{name}.summary.json", "w"))
