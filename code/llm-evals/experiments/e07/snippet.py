import sys; sys.path.insert(0, "../evals/lab")
import numpy as np, pandas as pd, arena

df = pd.read_csv("data/e07-votes.csv").sort_values("ts")   # 135,634 votes
a, b, y, models = arena.encode(df.model_a, df.model_b, df.winner)
M = len(models)

bt = arena.fit_bt(a, b, y, M)                       # order-free maximum likelihood
elo = arena.online_elo(a, b, y, M, k=4)             # replay votes in time order
shuffled = arena.online_elo(a, b, y, M, k=4, order=np.random.default_rng(0).permutation(len(y)))

i = models.index("gemini-2.5-pro")
print(round(bt[i]), round(elo[i]), round(shuffled[i]))
