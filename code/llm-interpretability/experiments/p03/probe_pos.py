# Part-of-speech probes per layer: linear and MLP, real labels and Hewitt & Liang control labels.
# Usage: probe_pos.py gpt2|qwen [layers comma-separated] [--quick]
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.probes import load_layer, train_probe, control_labels, bootstrap_ci, paired_bootstrap_diff

R = "/home/amit/.cache/vidyalai-scratch/p03/runs"
which = sys.argv[1]
A = f"{R}/acts/{which}"
info = json.load(open(f"{A}/pos_train/info.json"))
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] != "all" else list(range(info["n_layers"] + 1))
CTRL_SEEDS = [0, 1, 2]

pos = json.load(open(f"{R}/pos_data.json"))
def flat(split):
    forms, tags, sent = [], [], []
    for i, s in enumerate(pos[split]):
        for f, t, _, _ in s["words"]:
            forms.append(f); tags.append(t); sent.append(i)
    return forms, tags, np.array(sent)
ftr, ttr, _ = flat("train"); fte, tte, ste = flat("test")
TAGS = sorted(set(ttr) | set(tte)); T = {t: i for i, t in enumerate(TAGS)}
ytr = np.array([T[t] for t in ttr]); yte = np.array([T[t] for t in tte])
seen = np.array([f in set(ftr) for f in fte])

# baselines that never look at the model
maj = np.bincount(ytr).argmax()
from collections import Counter, defaultdict
cnt = defaultdict(Counter)
for f, y in zip(ftr, ytr):
    cnt[f][y] += 1
wid = np.array([cnt[f].most_common(1)[0][0] if f in cnt else maj for f in fte])
base = {"n_train_words": len(ytr), "n_test_words": len(yte), "n_test_sentences": int(ste.max() + 1),
        "n_tags": len(TAGS), "tags": TAGS, "frac_test_words_seen": float(seen.mean()),
        "majority": bootstrap_ci(yte == maj, ste), "word_identity": bootstrap_ci(wid == yte, ste),
        "word_identity_unseen": float((wid == yte)[~seen].mean()), "word_identity_seen": float((wid == yte)[seen].mean())}
ctrl = {}
for s in CTRL_SEEDS:
    m = control_labels(ftr + fte, ttr, seed=s)
    ctrl[s] = (np.array([T[m[f]] for f in ftr]), np.array([T[m[f]] for f in fte]))
print(json.dumps({k: v for k, v in base.items() if k != "tags"}), flush=True)

out = {"base": base, "layers": {}}
t0 = time.time()
for l in layers:
    Xtr = load_layer(f"{A}/pos_train", l); Xte = load_layer(f"{A}/pos_test", l)
    res = {}
    for kind in ["linear", "mlp"]:
        r = train_probe(Xtr, ytr, Xte, yte, kind=kind, n_classes=len(TAGS))
        ok = r["pred"] == yte
        res[kind] = {"acc": bootstrap_ci(ok, ste), "train_acc": r["train_acc"],
                     "acc_seen": float(ok[seen].mean()), "acc_unseen": float(ok[~seen].mean())}
        cs = []
        for s in CTRL_SEEDS:
            c = train_probe(Xtr, ctrl[s][0], Xte, ctrl[s][1], kind=kind, n_classes=len(TAGS), seed=s)
            cok = c["pred"] == ctrl[s][1]
            cs.append({"acc": float(cok.mean()), "train_acc": c["train_acc"],
                       "selectivity": paired_bootstrap_diff(ok, cok, ste, seed=s)})
        res[kind]["control"] = cs
        res[kind]["control_acc_mean"] = float(np.mean([c["acc"] for c in cs]))
        res[kind]["selectivity_mean"] = float(np.mean([c["selectivity"][0] for c in cs]))
    out["layers"][l] = res
    print(l, f"lin {res['linear']['acc'][0]:.4f} ctl {res['linear']['control_acc_mean']:.4f} | "
             f"mlp {res['mlp']['acc'][0]:.4f} ctl {res['mlp']['control_acc_mean']:.4f} | "
             f"unseen lin {res['linear']['acc_unseen']:.3f} | {time.time()-t0:.0f}s", flush=True)
tag = sys.argv[3] if len(sys.argv) > 3 else "all"
json.dump(out, open(f"{R}/pos_{which}_{tag}.json", "w"), indent=1)
