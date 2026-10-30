# Part-of-speech linear probes on the 88M model's residual stream, tracked over training checkpoints.
# Data: Part 3's UD English EWT v2.18 set (pos_data.json: 4,000 train sentences, all 2,077 test sentences).
# Each word is read at its last token. Layer l = residual stream entering block l (10 = last block's output).
# Usage: probe_track.py step,step,... ; resumable (one JSON per checkpoint).
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import dynamics as D
from lab.probes import train_probe, control_labels, bootstrap_ci
from tokenizer import load_production
OUT = "/home/amit/.cache/vidyalai-scratch/p10/runs/probe"; os.makedirs(OUT, exist_ok=True)
LAYERS = [0, 2, 4, 6, 8, 10]
pos = json.load(open("/home/amit/.cache/vidyalai-scratch/p03/runs/pos_data.json"))
tok = load_production()

def encode(split):
    seqs, where, tags, forms, sent = [], [], [], [], []
    for si, s in enumerate(pos[split]):
        enc = tok.encode(s["text"]); ids = [0] + enc.ids; starts = [o[0] for o in enc.offsets]
        for f, t, a, b in s["words"]:
            last = max(i for i, st in enumerate(starts) if st < b)
            where.append((si, last + 1)); tags.append(t); forms.append(f); sent.append(si)
        seqs.append(ids)
    return seqs, where, tags, forms, np.array(sent)

tr, te = encode("train"), encode("test")
TAGS = sorted(set(tr[2]) | set(te[2])); T = {t: i for i, t in enumerate(TAGS)}
ytr = np.array([T[t] for t in tr[2]]); yte = np.array([T[t] for t in te[2]])
seen = np.array([f in set(tr[3]) for f in te[3]])
cm = control_labels(tr[3] + te[3], tr[2], seed=0)
ctr = np.array([T[cm[f]] for f in tr[3]]); cte = np.array([T[cm[f]] for f in te[3]])

@torch.no_grad()
def acts(model, data, bs=64):
    seqs, where = data[0], data[1]
    res = {l: [] for l in LAYERS}
    by_sent = {}
    for k, (si, p) in enumerate(where): by_sent.setdefault(si, []).append((k, p))
    order = np.zeros(len(where), dtype=np.int64); rows = 0
    store = {}
    hooks = [model.blocks[l].register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(l, a[0])) for l in LAYERS if l < model.cfg.n_layer]
    hooks.append(model.blocks[-1].register_forward_hook(lambda m, a, o: store.__setitem__(model.cfg.n_layer, o)))
    for b in range(0, len(seqs), bs):
        chunk = seqs[b:b + bs]; Lm = max(len(s) for s in chunk)
        x = torch.zeros(len(chunk), Lm, dtype=torch.long)
        for i, s in enumerate(chunk): x[i, :len(s)] = torch.tensor(s)
        model(x.cuda())
        idx_s, idx_p, ks = [], [], []
        for i in range(len(chunk)):
            for k, p in by_sent.get(b + i, []): idx_s.append(i); idx_p.append(p); ks.append(k)
        for l in LAYERS: res[l].append(store[l][idx_s, idx_p].half().cpu())
        order[rows:rows + len(ks)] = ks; rows += len(ks)
    for h in hooks: h.remove()
    inv = np.argsort(order)
    return {l: torch.cat(v)[inv].float().numpy() for l, v in res.items()}

steps = [int(s) for s in sys.argv[1].split(",")]
t0 = time.time()
for step in steps:
    f = f"{OUT}/{step:05d}.json"
    if os.path.exists(f): continue
    m = D.load(f"{D.RUNS['main88'].path}/model_{step:05d}.pt", D.RUNS["main88"].cfg)
    Atr, Ate = acts(m, tr), acts(m, te); del m
    out = {"step": step, "layers": {}}
    for l in LAYERS:
        r = train_probe(Atr[l], ytr, Ate[l], yte, kind="linear", n_classes=len(TAGS))
        ok = r["pred"] == yte
        c = train_probe(Atr[l], ctr, Ate[l], cte, kind="linear", n_classes=len(TAGS))
        out["layers"][l] = {"acc": bootstrap_ci(ok, te[4]), "unseen": float(ok[~seen].mean()), "seen": float(ok[seen].mean()),
                            "control": float((c["pred"] == cte).mean()), "train_acc": r["train_acc"]}
    json.dump(out, open(f, "w"), indent=1)
    print(step, {l: (round(v["acc"][0], 3), round(v["unseen"], 3), round(v["control"], 3)) for l, v in out["layers"].items()},
          f"{time.time()-t0:.0f}s", flush=True)
print("n_train", len(ytr), "n_test", len(yte), "unseen frac", 1 - seen.mean())
