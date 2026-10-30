"""Audit of Part 1's headline: DLA vs mean ablation on 168 landmark facts, under new templates.
Usage: python audit_landmark.py gpt2|qwen   -> runs/landmark_<model>.json"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")   # read-only import of Part 1's fact table
from lab import hooks as H, replicate as R
from facts import FACTS

which = sys.argv[1]
torch.set_grad_enabled(False)
name, proc = {"gpt2": ("gpt2", True), "qwen": ("Qwen/Qwen2.5-0.5B", False)}[which]
t0 = time.time()
model = H.load(name, dtype=torch.float32, processing=proc)
START = 1   # leave the first position alone (GPT-2's BOS; Qwen's first word, as in Part 1)
facts = [f for f in FACTS if f[1] not in ("Agra", "Granada", "Krakow", "Pisa", "Mecca")]
TEMPLATES = [("original", "{} is located in the city of", False),
             ("landmark", "{} is a famous landmark in the city of", False),
             ("travel", "If you want to see {}, you have to travel to the city of", True)]
out = {"model": model.setup_info, "start": START}
for tag, tpl, low in TEMPLATES:
    t1 = time.time()
    keep, prompts, cities, cid, ans = R.landmark_prompts(model, facts, tpl, lower_the=low)
    r = R.layer_audit(model, prompts, cid, ans, tpl, START)
    labs = r["labels"]
    ag = R.rank_agreement(r["dla"], -r["d_mean"])
    o = {"template": tpl, "n": len(prompts), "n_cities": len(cities),
         "top1_vocab": R.boot_mean(r["top1"].astype(float)), "top1_cities": R.boot_mean(r["topc"].astype(float)),
         "ld": R.boot_mean(r["ld"]), "agree": ag,
         "rank_dla": [labs[i] for i in np.argsort(-r["dla"].mean(1))[:6]],
         "rank_mean_ablation": [labs[i] for i in np.argsort(r["d_mean"].mean(1))[:6]],
         "per_comp": {l: {"dla": R.boot_mean(r["dla"][i], n_boot=4000), "cost": R.boot_mean(-r["d_mean"][i], n_boot=4000)}
                      for i, l in enumerate(labs)},
         "seconds": round(time.time() - t1, 1)}
    out[tag] = o
    np.savez_compressed(f"/home/amit/.cache/vidyalai-scratch/p12/runs/landmark_{which}_{tag}.npz", **{k: v for k, v in r.items() if k != "labels"})
    pc = o["per_comp"]
    print(tag, o["n"], "acc", round(o["top1_cities"][0], 3), "ld", [round(x, 2) for x in o["ld"]],
          "mlp0 cost", [round(x, 2) for x in pc["mlp0"]["cost"]], "rho", [round(x, 2) for x in ag["spearman"]],
          "top5", ag["top5_overlap"], "dla", o["rank_dla"][:4], "abl", o["rank_mean_ablation"][:4], f"{o['seconds']}s", flush=True)
out["seconds"] = round(time.time() - t0, 1)
out["gpu_peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/p12/runs/landmark_{which}.json", "w"), indent=1)
print("done", out["seconds"], "s", out["gpu_peak_mib"], "MiB")
