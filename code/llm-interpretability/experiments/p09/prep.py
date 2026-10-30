"""Build the instruction splits for Part 9 (kept local; harmful text is never committed).
harmful train/val: AdvBench (MIT), minus any item more than 70% similar (difflib) to a JBB item; harmful test: JBB-Behaviors harmful (MIT, 100)
harmless train/val/test: Alpaca (CC BY-NC 4.0) instructions with no input field; benign look-alikes: JBB benign (MIT, 100)"""
import csv, json, os, random, urllib.request
import pyarrow.parquet as pq
D = "/home/amit/.cache/vidyalai-scratch/interp/data/p09"
# The prompt sets are downloaded at run time and never committed (about 24.4 MB, almost all of it Alpaca).
SRC = {"jbb_harmful.csv": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
       "jbb_benign.csv": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv",
       "advbench_harmful_behaviors.csv": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
       "alpaca_train.parquet": "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}
os.makedirs(D, exist_ok=True)
for f, url in SRC.items():
    if not os.path.exists(f"{D}/{f}"):
        urllib.request.urlretrieve(url, f"{D}/{f}")
jbb_h = list(csv.DictReader(open(f"{D}/jbb_harmful.csv")))
jbb_b = list(csv.DictReader(open(f"{D}/jbb_benign.csv")))
adv = [r["goal"] for r in csv.DictReader(open(f"{D}/advbench_harmful_behaviors.csv"))]
jbb_goals = {r["Goal"].strip().lower() for r in jbb_h}
import difflib
adv = [g for g in adv if max(difflib.SequenceMatcher(None, g.lower(), j).ratio() for j in jbb_goals) <= 0.7]
print("advbench after removing JBB overlap:", len(adv))
rng = random.Random(0); rng.shuffle(adv)
t = pq.read_table(f"{D}/alpaca_train.parquet", columns=["instruction", "input"]).to_pylist()
alp = sorted({r["instruction"].strip() for r in t if not r["input"].strip()})
print("alpaca no-input instructions:", len(alp))
rng.shuffle(alp)
out = {"harmful_train": adv[:128], "harmful_val": adv[128:160],
       "harmful_test": [r["Goal"] for r in jbb_h], "harmful_test_cat": [r["Category"] for r in jbb_h],
       "harmless_train": alp[:128], "harmless_val": alp[128:160], "harmless_test": alp[160:260],
       "benign_test": [r["Goal"] for r in jbb_b], "benign_test_cat": [r["Category"] for r in jbb_b]}
json.dump(out, open(f"{D}/splits.json", "w"), indent=1)
print({k: len(v) for k, v in out.items()})
