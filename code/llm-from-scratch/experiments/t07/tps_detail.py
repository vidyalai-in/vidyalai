import json, numpy as np
recs=[json.loads(l) for l in open("runs/main88/log.jsonl")]; tr=[r for r in recs if "loss" in r]
dt=np.array([r["dt"] for r in tr]); tps=np.array([r["tok_per_s"] for r in tr]); el=np.array([r["elapsed"] for r in tr])
print("hist tok/s:", np.histogram(tps[1:], bins=[25000,26000,27000,28000,28500,29000,29500,30000,31000,32000,33000,34000]))
pair=dt[1:-1:2]+dt[2::2]; print("dt sd", dt[1:].std().round(3), "pair-sum sd/2", (pair.std()/2).round(3), "pair mean tok/s", round(2*524288/pair.mean()))
r=np.corrcoef(dt[1:-1], dt[2:])[0,1]; print("lag-1 autocorr of dt", round(r,3))
for h in range(17):
    m=(el>=h*3600)&(el<(h+1)*3600)
    print(h, "steps", m.sum(), "median tok/s", int(np.median(tps[m])), "tokens/s over the hour (step time)", int(m.sum()*524288/dt[m].sum()))
# 10-step blocks
blk=[(i, round(10*524288/dt[i:i+10].sum())) for i in range(1,3351,10)]
v=np.array([b for _,b in blk]); print("10-step block tok/s: median", int(np.median(v)), "p5", int(np.percentile(v,5)), "p95", int(np.percentile(v,95)), "min", v.min(), "max", v.max())
print(tps[2940:2956], dt[2940:2956])
