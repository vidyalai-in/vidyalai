# Collect file sizes, llama-perplexity, KL divergence and llama-bench results into post/quant_summary.json.
import json, math, os, re
T = "/home/amit/.cache/vidyalai-scratch/t15"
ref = {r["ckpt"].split("/")[-2] + ("-bf16" if r["bf16"] else ""): r for r in json.load(open(f"{T}/post/ppl_reference.json"))}
tok, nbytes = ref["main88"]["tokens_scored"], ref["main88"]["bytes_scored"]
bench = {}
for f in ("bench_run1/bench_t8.json", "bench_run1/bench_t16.json", "bench_t8.json", "bench_t16.json"):
    run = "run1" if "run1" in f else "run2"
    if os.path.exists(f"{T}/post/{f}"):
        for r in json.load(open(f"{T}/post/{f}")):
            name = os.path.basename(r["model_filename"])[:-5]
            kind = "pp" if r["n_prompt"] else "tg"
            bench.setdefault(name, {})[f"{kind}_t{r['n_threads']}_{run}"] = (round(r["avg_ts"], 1), round(r["stddev_ts"], 1))
out = []
for m in ("base88", "chat88"):
    for t in ("f32", "f16", "q8_0", "q4_k_m"):
        name = f"{m}-{t}"
        log = open(f"{T}/post/ppl/{name}.log").read()
        ppl, err = map(float, re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", log).groups())
        row = {"file": name, "bytes": os.path.getsize(f"{T}/gguf/{name}.gguf"), "ppl": ppl, "ppl_err": err,
               "bpb": math.log(ppl) * tok / math.log(2) / nbytes, "bench": bench.get(name)}
        if t != "f32":
            k = open(f"{T}/post/ppl/{name}-kld.log").read()
            row["kld_mean"] = float(re.search(r"Mean\s+KLD:\s+([\d.]+)", k).group(1))
            row["kld_99"] = float(re.search(r"99.0%\s+KLD:\s+([\d.]+)", k).group(1))
            row["same_top"] = float(re.search(r"Same top p:\s+([\d.]+)", k).group(1))
        q = f"{T}/gguf/{name}.log"
        if os.path.exists(q) and "BPW" in open(q).read():
            row["bpw"] = float(re.findall(r"\(([\d.]+) BPW\)", open(q).read())[-1])
        out.append(row)
        print(json.dumps(row))
json.dump({"reference": ref, "gguf": out}, open(f"{T}/post/quant_summary.json", "w"), indent=1)
