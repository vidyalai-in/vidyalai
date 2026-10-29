# Does the truth direction transfer to a second template? Threshold-free check with AUROC.
# For each fold: mass-mean direction from the training cities (template 1), scored on held-out
# cities in template 1 and template 2 ("{city} is a city in {country}.").
import json, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.probes import load_layer, mass_mean_direction
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
R = "/home/amit/.cache/vidyalai-scratch/p03/runs"
items = json.load(open(f"{R}/truth_data.json"))
y = np.array([it["label"] for it in items]); city = np.array([it["city"] for it in items])
folds = list(GroupKFold(n_splits=5).split(y, y, groups=city))
out = {}
for which in ["gpt2", "qwen"]:
    nl = json.load(open(f"{R}/acts/{which}/truth_cities/info.json"))["n_layers"]
    out[which] = {}
    for l in range(nl + 1):
        X = load_layer(f"{R}/acts/{which}/truth_cities", l); Xa = load_layer(f"{R}/acts/{which}/truth_alt", l)
        s, sa = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in folds:
            th = mass_mean_direction(X[tr], y[tr]); s[te] = X[te] @ th; sa[te] = Xa[te] @ th
        # accuracy on template 2 if the threshold is re-set at the midpoint of template-2 class means
        out[which][l] = {"auc_same": float(roc_auc_score(y, s)), "auc_alt": float(roc_auc_score(y, sa))}
    print(which, {l: (round(v["auc_same"], 3), round(v["auc_alt"], 3)) for l, v in out[which].items()})
json.dump(out, open(f"{R}/transfer_auc.json", "w"), indent=1)
