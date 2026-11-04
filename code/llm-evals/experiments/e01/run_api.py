"""e01 API runs. Haiku 5.5: every item of the four sets as full questions; choices-only on
ARC-Challenge, MMLU and MMLU-Pro; NOTA variants on MMLU. Sonnet 5.5: full questions on a
200-item random subset (seed 0) of ARC-Easy, MMLU and MMLU-Pro. Thinking off, default sampling, one sample per item."""
import json, random, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
OUT = "/home/amit/.cache/vidyalai-scratch/e01/runs"
SETS = {"arce": "e01-arc-easy.json", "arcc": "e01-arc-challenge.json", "mmlu": "mmlu-1140.json",
        "pro": "e01-mmlu-pro-840.json"}
KIND = {"full": "f", "choices_only": "co", "nota_correct": "nc", "nota_wrong": "nw"}
data = {k: V.load(f"{D}/{v}") for k, v in SETS.items()}

def jobs_for(model, plan):
    jobs = {}
    for s, kind, idxs in plan:
        for i in idxs:
            jobs[f"{s}-{KIND[kind]}-{i}"] = V.request(data[s][i], kind, model)
    return jobs

def subset(s, n=200):
    idx = list(range(len(data[s])))
    random.Random(0).shuffle(idx)
    return sorted(idx[:n])

which = sys.argv[1]
if which == "haiku":
    allidx = {s: range(len(data[s])) for s in data}
    plan = [(s, "full", allidx[s]) for s in data]
    plan += [(s, "choices_only", allidx[s]) for s in ("arcc", "mmlu", "pro")]
    plan += [("mmlu", "nota_correct", allidx["mmlu"]), ("mmlu", "nota_wrong", allidx["mmlu"])]
    jobs = jobs_for("claude-haiku-5-5", plan)
else:
    jobs = jobs_for("claude-sonnet-5-5", [(s, "full", subset(s)) for s in ("arce", "mmlu", "pro")])
print(which, len(jobs), flush=True)
print(V.run_batch("e01", jobs, f"{OUT}/{which}.jsonl", note=f"e01 {which} batch", count=(which == "sonnet")), flush=True)
