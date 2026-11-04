import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import arena

def load(cols=()):
    base = ["ts", "model_a", "model_b", "winner"]
    df = pd.read_csv("/home/amit/.cache/vidyalai-scratch/e07/data/e07-votes.csv", usecols=list(dict.fromkeys(base + list(cols))))
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    cnt = pd.concat([df.model_a, df.model_b]).value_counts()
    small = set(cnt[cnt < 100].index)
    return df[~df.model_a.isin(small) & ~df.model_b.isin(small)].reset_index(drop=True)
