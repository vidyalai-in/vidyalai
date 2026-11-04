"""Part 4, experiment 1: one open model, every scoring method, format and few-shot draw on mmlu-1140.
Usage: run_open.py <hf model id> [--check] [--limit-subjects N]"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq

mid = sys.argv[1]
check = "--check" in sys.argv
nsub = int(sys.argv[sys.argv.index("--limit-subjects") + 1]) if "--limit-subjects" in sys.argv else None
ITEMS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
SHOTS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e04-shots.json"))
subjects = sorted({it["subject"] for it in ITEMS})[:nsub]
by_subj = {s: [i for i, it in enumerate(ITEMS) if it["subject"] == s] for s in subjects}

t0 = time.time()
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
sc = mcq.LogProbScorer(model, tok)
print(f"loaded {mid} in {time.time()-t0:.1f}s", flush=True)

def shots(subj, draw, k):
    return SHOTS[subj][draw][:k] if k else []

CONDS = []
for f in mcq.FORMATS:
    CONDS += [("letter", f, "dev", 0), ("letter", f, "dev", 5)]
CONDS += [("letter", "mmlu", d, k) for d, k in [("dev", 1), ("s1", 1), ("s2", 1), ("s1", 5), ("s2", 5)]]
CONDS += [("cloze", "mmlu", "dev", 0), ("cloze", "mmlu", "dev", 5)]
CONDS += [("gen", "mmlu", "dev", 0), ("gen", "mmlu", "dev", 5)]
CONDS += [("letternc", "mmlu", "dev", 5)]  # same as letter-mmlu-dev5 but without the shared KV cache (bf16 check)

if check:
    # compare cached, batched scoring with a plain full-sequence forward pass
    fmt = mcq.FORMATS["mmlu"]
    for subj in subjects[:2]:
        idx = by_subj[subj][:3]
        for k in (0, 5):
            parts = [mcq.prompt(fmt, ITEMS[i], shots(subj, "dev", k)) for i in idx]
            got = sc.label_scores(parts[0][0], [o for _, o in parts], fmt)
            cz = sc.cloze_scores(mcq.prompt(fmt, ITEMS[idx[0]], shots(subj, "dev", k), False)[0],
                                 [mcq.prompt(fmt, ITEMS[i], shots(subj, "dev", k), False)[1] for i in idx], [ITEMS[i] for i in idx])
            for j, (sh, o) in enumerate(parts):
                full = sc.bos + sc.enc(sh + o)
                with torch.no_grad():
                    hh = model.model(input_ids=torch.tensor([full], device="cuda")).last_hidden_state[0, -1:]
                    lp = sc._logprobs_at(hh.unsqueeze(0), [(0, 0)])[0]
                ref = [lp[sc.enc(o + fmt.target(c))[-1]].item() for c in range(4)]
                print(subj, k, "letter cached", [round(x, 3) for x in got[j]], "plain", [round(x, 3) for x in ref])
                # cloze reference for option 0
                sh2, o2 = mcq.prompt(fmt, ITEMS[idx[j]], shots(subj, "dev", k), False)
                ctx = sc.bos + sc.enc(sh2 + o2)
                fullc = sc.bos + sc.enc(sh2 + o2 + " " + str(ITEMS[idx[j]]["choices"][0]).strip())
                with torch.no_grad():
                    hh = model.model(input_ids=torch.tensor([fullc], device="cuda")).last_hidden_state
                    lpc = sc._logprobs_at(hh, [(0, t) for t in range(len(fullc))])
                refc = sum(lpc[t - 1, fullc[t]].item() for t in range(len(ctx), len(fullc)))
                print("   cloze opt0 cached", round(cz[j][0][0], 3), "plain", round(refc, 3))
            if k == 5:
                g = sc.generate(parts[0][0], [o for _, o in parts])
                print("   gen", g)
    sys.exit()

tag = mid.split("/")[-1]
path = f"/home/amit/.cache/vidyalai-scratch/e04/runs/open-{tag}.json"
res = json.load(open(path)) if os.path.exists(path) else {"model": mid, "conds": {}, "timing": {}}
res.setdefault("gpu_s_parts", [])
for kind, f, draw, k in CONDS:
    name = f"{kind}-{f}-{draw}{k}"
    if name in res["conds"]:
        continue  # resume: this condition is already done
    fmt = mcq.FORMATS[f]
    t1 = time.time()
    out = [None] * len(ITEMS)
    for subj in subjects:
        idx = by_subj[subj]
        parts = [mcq.prompt(fmt, ITEMS[i], shots(subj, draw, k), kind != "cloze") for i in idx]
        if kind == "letternc":
            parts = [("", a + b) for a, b in parts]
        assert len({p[0] for p in parts}) == 1
        owns = [o for _, o in parts]
        if kind in ("letter", "letternc"):
            r = sc.label_scores(parts[0][0], owns, fmt)
        elif kind == "cloze":
            r = sc.cloze_scores(parts[0][0], owns, [ITEMS[i] for i in idx])
        else:
            r = sc.generate(parts[0][0], owns)
        for i, x in zip(idx, r):
            out[i] = x
    res["conds"][name] = out
    res["timing"][name] = round(time.time() - t1, 1)
    if kind in ("letter", "letternc"):
        acc = sum(max(range(4), key=lambda c: out[i][c]) == ITEMS[i]["answer"] for i in range(len(ITEMS)) if out[i]) / sum(1 for x in out if x)
        print(f"{name:24s} {res['timing'][name]:6.1f}s acc {acc:.3f}", flush=True)
    else:
        print(f"{name:24s} {res['timing'][name]:6.1f}s", flush=True)
    json.dump(res, open(path, "w"))
res["gpu_s_parts"].append(round(time.time() - t0, 1))
res["gpu_s"] = round(sum(res["gpu_s_parts"]), 1)
res["max_mem_gb"] = max(res.get("max_mem_gb", 0), round(torch.cuda.max_memory_allocated() / 1e9, 2))
json.dump(res, open(path, "w"))
print("total", res["gpu_s"], "s, peak", res["max_mem_gb"], "GB", flush=True)
