"""Tables from post/results/*.json: bpb, accuracy with 95% intervals, chance levels, a continuous metric."""
import json, math, glob, os
R = "post/results"
def load(t): return json.load(open(f"{R}/{t}.json"))
def acc(items, norm=False):
    hits = [max(range(len(r["lls"])), key=lambda c: r["lls"][c] / (r["chars"][c] if norm else 1)) == r["gold"] for r in items]
    p = sum(hits) / len(hits); return 100 * p, 196 * math.sqrt(p * (1 - p) / len(hits))
def pcorrect(items):   # probability mass on the gold option, options renormalized (softmax over summed log-likelihoods)
    out = []
    for r in items:
        m = max(r["lls"]); z = sum(math.exp(x - m) for x in r["lls"]); out.append(math.exp(r["lls"][r["gold"]] - m) / z)
    return 100 * sum(out) / len(out)
def chance(items): return 100 * sum(1 / len(r["lls"]) for r in items) / len(items)
models = ["s11", "s15", "s22", "s33", "s47", "main88", "pythia70m", "gpt2_fp32", "pythia160m", "smollm2"]
g = load("main88")
for t in ["hellaswag_0", "arc_easy_0", "piqa_0"]:
    print(t, "chance", round(chance(g["tasks"][t]["items"]), 2), "n", len(g["tasks"][t]["items"]))
print(f"{'model':11} {'params':>12} {'bpbFW':>6} {'bpbWT':>6} {'pplFW':>6} {'pplWT':>6} {'B/tFW':>5} {'B/tWT':>5} | HS acc, norm | ARC acc, norm | PIQA acc, norm")
for m in models:
    d = load(m); b = d["bpb"]
    row = f"{m:11} {d['params']:12,} {b['fineweb']['bpb']:6.4f} {b['wikitext']['bpb']:6.4f} {b['fineweb']['perplexity']:6.2f} {b['wikitext']['perplexity']:6.2f} {b['fineweb']['bytes_per_token']:5.3f} {b['wikitext']['bytes_per_token']:5.3f} |"
    for t in ["hellaswag_0", "arc_easy_0", "piqa_0"]:
        it = d["tasks"][t]["items"]; a, ac = acc(it); n, nc = acc(it, True)
        row += f" {a:5.2f}±{ac:.2f} {n:5.2f}±{nc:.2f} |"
    print(row)
print("\nover training (88M), continuous metric = mean prob on correct option (%)")
for m in ["step250", "step500", "step1000", "step2000", "main88"]:
    d = load(m); row = f"{m:9}"
    for t in ["hellaswag_0", "arc_easy_0", "piqa_0"]:
        it = d["tasks"][t]["items"]; a, ac = acc(it); n, nc = acc(it, True)
        row += f" | {t[:4]} acc {a:5.2f}±{ac:.2f} norm {n:5.2f}±{nc:.2f} p {pcorrect(it):5.2f}"
    print(row)
print("\nover size")
for m in ["s11", "s15", "s22", "s33", "s47", "main88"]:
    d = load(m); row = f"{m:9}"
    for t in ["hellaswag_0", "arc_easy_0", "piqa_0"]:
        it = d["tasks"][t]["items"]; row += f" | {t[:4]} p {pcorrect(it):5.2f}"
    print(row)
print("\nBF16 autocast vs FP32 (HF models)")
for m in ["gpt2", "pythia70m", "pythia160m"]:
    a = json.load(open(f"{R}/bf16/{m}.json")); f = load("gpt2_fp32" if m == "gpt2" else m)
    print(m, "bpb FW", round(f["bpb"]["fineweb"]["bpb"], 4), "->", round(a["bpb"]["fineweb"]["bpb"], 4),
          "WT", round(f["bpb"]["wikitext"]["bpb"], 4), "->", round(a["bpb"]["wikitext"]["bpb"], 4),
          *[f"{t}: {acc(f['tasks'][t]['items'], t.startswith('hella'))[0]:.2f}->{acc(a['tasks'][t]['items'], t.startswith('hella'))[0]:.2f}" for t in ["hellaswag_0", "arc_easy_0", "piqa_0"]])
fs = load("main88_fewshot")
for t in fs["tasks"]: print("fewshot", t, acc(fs["tasks"][t]["items"]), acc(fs["tasks"][t]["items"], True))
for m in models + ["step250", "step500", "step1000", "step2000"]:
    d = load(m); print(m, {k: round(v, 1) for k, v in d["seconds"].items()}, "peak GiB", round(d["peak_gib"], 2))
