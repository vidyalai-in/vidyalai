# Part 7: data lines for the diagrams/t07-*.py chart generators, from the run's logs (CPU only).
import json, numpy as np
D = "/home/amit/.cache/vidyalai-scratch/t07"
recs = [json.loads(l) for l in open(f"{D}/runs/main88/log.jsonl")]
tr = [r for r in recs if "loss" in r]
out = {}
out["TRAIN"] = [round(r["loss"], 3) for r in tr]
out["GN"] = [round(r["grad_norm"], 3) for r in tr]
out["VAL"] = [(r["step"], round(r["val_loss"], 4)) for r in recs if "val_loss" in r]
dt = np.array([r["dt"] for r in tr]); el = np.array([r["elapsed"] for r in tr])
out["TPS10"] = [(round(float(el[i + 9]) / 3600, 3), int(round(10 * 524288 / dt[i:i + 10].sum()))) for i in range(1, 3351, 10)]
out["SMI"] = json.load(open(f"{D}/post/smi_bins.json"))
wn = json.load(open(f"{D}/post/weight_norms.json"))
out["NORMS"] = [(w["step"], round(w["emb"], 3), round(w["matrices_excl_emb"], 3), round(w["down"], 3), round(w["final_norm_gain_mean"], 4)) for w in wn]
for k, v in out.items():
    open(f"{D}/post/chart_{k}.txt", "w").write(f"{k}={json.dumps(v).replace(' ', '')}\n".replace("[[", "[(").replace("]]", ")]").replace("],[", "),("))
    print(k, len(v))
