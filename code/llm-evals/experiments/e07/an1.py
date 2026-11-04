import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena
o = json.load(open("runs/exp1.json"))
d = pd.DataFrame({k: o[k] for k in o if isinstance(o[k], list) and len(o[k]) == o["n_models"]})
d["first"] = d.first_vote.str[:10]
d["rk_bt"] = arena.ranks(d.bt)
for c in ["elo_chrono_k4", "elo_reverse_k4", "elo_chrono_k32", "elo_reverse_k32", "elo_perm_k4_mean", "bt_drop_ties", "rao_kupper", "elo_chrono_k16"]:
    d["rk_" + c] = arena.ranks(d[c])
    print(f"{c:22s} tau={arena.kendall_tau(d.bt, d[c]):.3f} max|rank shift|={np.abs(d['rk_'+c]-d.rk_bt).max()} mean|shift|={np.abs(d['rk_'+c]-d.rk_bt).mean():.2f} sd={np.std(d[c]):.1f} max|dr|={np.abs(d[c]-d.bt).max():.1f}")
print("BT sd", np.std(d.bt), "range", d.bt.max() - d.bt.min(), "theta", o["rk_theta"])
pd.set_option("display.width", 250)
cols = ["models", "votes_per_model", "first", "bt", "bt_lo", "bt_hi", "bt_arena_rank", "rk_bt", "elo_chrono_k4", "rk_elo_chrono_k4", "elo_reverse_k4", "elo_perm_k4_lo", "elo_perm_k4_hi", "elo_perm_k4_rank_min", "elo_perm_k4_rank_max", "elo_chrono_k32", "rk_elo_chrono_k32", "elo_perm_k32_lo", "elo_perm_k32_hi", "elo_perm_k32_rank_min", "elo_perm_k32_rank_max", "rk_bt_drop_ties", "rk_rao_kupper"]
print(d.sort_values("bt", ascending=False)[cols].round(0).to_string())
print(o["outcomes"], o["date_range"])
w = (d.bt_hi - d.bt_lo) / 2; print("CI half-width median", w.median(), "min", w.min(), "max", w.max())
w4 = (d.elo_perm_k4_hi - d.elo_perm_k4_lo)/2; w32=(d.elo_perm_k32_hi - d.elo_perm_k32_lo)/2
print("perm k4 half-range median", w4.median(), "max", w4.max(), " k32 median", w32.median(), "max", w32.max())
