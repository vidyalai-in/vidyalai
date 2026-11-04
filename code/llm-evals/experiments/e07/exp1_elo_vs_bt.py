"""Experiment 1: online Elo under different vote orders and K factors vs Bradley-Terry, on the
135,634 votes of lmarena-ai/arena-human-preference-140k (April 17 to July 24, 2025)."""
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena
t0 = time.time()
df = pd.read_csv("data/e07-votes.csv", usecols=["ts", "model_a", "model_b", "winner"]).sort_values("ts", kind="stable").reset_index(drop=True)
cnt = pd.concat([df.model_a, df.model_b]).value_counts()
small = set(cnt[cnt < 100].index)  # qwen3-coder-480b-a35b-instruct: 2 votes
df = df[~df.model_a.isin(small) & ~df.model_b.isin(small)].reset_index(drop=True)
a, b, y, models = arena.encode(df.model_a, df.model_b, df.winner)
M, N = len(models), len(y)
nv = np.bincount(a, minlength=M) + np.bincount(b, minlength=M)
first = pd.concat([df[["ts"]].assign(m=df.model_a), df[["ts"]].assign(m=df.model_b)]).groupby("m").ts.min()
out = {"n_votes": N, "dropped_models": sorted(small), "n_models": M, "models": models, "votes_per_model": nv.tolist(),
       "first_vote": [first[m] for m in models],
       "outcomes": df.winner.value_counts().to_dict(), "date_range": [df.ts.iloc[0], df.ts.iloc[-1]]}

bt = arena.fit_bt(a, b, y, M); out["bt"] = bt.tolist()
boot = arena.bootstrap_bt(a, b, y, M, n_boot=1000, seed=0)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
out["bt_lo"], out["bt_hi"] = lo.tolist(), hi.tolist()
out["bt_arena_rank"] = arena.arena_rank(lo, hi).tolist()
np.save("runs/exp1_boot.npy", boot)
print("BT done", time.time() - t0, flush=True)
out["bt_drop_ties"] = arena.fit_bt(a, b, y, M, ties="drop").tolist()
rk, theta = arena.rao_kupper(a, b, y, M); out["rao_kupper"], out["rk_theta"] = rk.tolist(), theta

for k in [4, 8, 16, 32]:
    out[f"elo_chrono_k{k}"] = arena.online_elo(a, b, y, M, k=k).tolist()
    out[f"elo_reverse_k{k}"] = arena.online_elo(a, b, y, M, k=k, order=np.arange(N)[::-1]).tolist()
rng = np.random.default_rng(1)
for k in [4, 32]:
    orders = np.stack([rng.permutation(N) for _ in range(200)])
    R = arena.online_elo_many(a, b, y, M, orders, k=k)
    np.save(f"runs/exp1_perm_k{k}.npy", R)
    out[f"elo_perm_k{k}_mean"] = R.mean(0).tolist()
    out[f"elo_perm_k{k}_lo"] = np.percentile(R, 2.5, 0).tolist()
    out[f"elo_perm_k{k}_hi"] = np.percentile(R, 97.5, 0).tolist()
    rks = np.stack([arena.ranks(r) for r in R])
    out[f"elo_perm_k{k}_rank_min"] = rks.min(0).tolist(); out[f"elo_perm_k{k}_rank_max"] = rks.max(0).tolist()
    print("perm", k, time.time() - t0, flush=True)
json.dump(out, open("runs/exp1.json", "w"))
print("done", time.time() - t0)
