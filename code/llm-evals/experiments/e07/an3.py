import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena
o = json.load(open("runs/exp3.json"))
d = pd.DataFrame({k: o[k] for k in ["models","bt","sc_both","sc_length","sc_markdown","sc_lo","sc_hi","mean_tokens","mean_markdown","se_session","se_vote"]})
for c in ["bt","sc_both","sc_length","sc_markdown"]: d["rk_"+c]=arena.ranks(d[c])
d["move"]=d.rk_bt-d.rk_sc_both
print("gammas both", np.round(o["gamma_both"],3), "lo", np.round(o["gamma_lo"],3), "hi", np.round(o["gamma_hi"],3))
print("gamma length only", o["gamma_length"], "markdown only", o["gamma_markdown"])
for c in ["sc_both","sc_length","sc_markdown"]:
    print(c, "tau", round(arena.kendall_tau(d.bt,d[c]),3), "max shift", np.abs(d.rk_bt-d["rk_"+c]).max(), "mean", np.abs(d.rk_bt-d["rk_"+c]).mean().round(2), "sd", np.std(d[c]).round(1))
print("corr(move, mean_tokens)", np.corrcoef(d.bt-d.sc_both, d.mean_tokens)[0,1], "corr md", np.corrcoef(d.bt-d.sc_both, d.mean_markdown)[0,1])
pd.set_option("display.width",250)
print(d.sort_values("bt",ascending=False)[["models","bt","rk_bt","sc_both","sc_lo","sc_hi","rk_sc_both","move","rk_sc_length","rk_sc_markdown","mean_tokens","mean_markdown"]].round(1).to_string())
print("longer wins", o["longer_wins_share"], o["longer_wins_share"][0]/o["longer_wins_share"][1])
print(o["mix"], o["N"])
for k,v in o["subsets"].items():
    r=np.array(v["bt"]); n=np.array(v["votes"]); m=n>=300
    rk_all=arena.ranks(d.bt[m].values); rk_s=arena.ranks(r[m])
    sh=rk_s-rk_all; i=np.argmax(np.abs(sh))
    print(k, v["n"], "models>=300", m.sum(), "tau", round(arena.kendall_tau(d.bt[m], r[m]),3), "max shift", np.abs(sh).max(), np.array(o["models"])[m][i], rk_all[i], "->", rk_s[i])
r=np.array(o["se_session"])/np.array(o["se_vote"]); print("session/vote SE ratio median", np.median(r), r.min(), r.max(), "design effect", np.median(r**2))
