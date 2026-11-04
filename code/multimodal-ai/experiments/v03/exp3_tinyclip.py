"""Exp 3: a tiny CLIP trained from scratch on synthetic shape scenes.
Usage: python exp3_tinyclip.py render            (CPU: writes the image pool)
       python exp3_tinyclip.py run <tag> <loss> <batch> <tau|learned> <examples> <seed>...
"""
import json, math, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import clip_utils as cu

OUT = "/home/amit/.cache/vidyalai-scratch/v03"
HELD = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]  # (shape, colour) pairs never seen in training
COMBOS = cu.all_captions()                                # 6*6*2*4 = 288
CID = {c: i for i, c in enumerate(COMBOS)}
WORDS = sorted({w for c in COMBOS for w in cu.caption(*c).split()})
TOK = torch.tensor([[WORDS.index(w) for w in cu.caption(*c).split()] for c in COMBOS])  # [288, 8]

if sys.argv[1] == "render":
    g = np.random.default_rng(0)
    seen = [c for c in COMBOS if (c[0], c[1]) not in HELD]
    unseen = [c for c in COMBOS if (c[0], c[1]) in HELD]
    def make(pool, n, seed):
        attrs = np.array([pool[i] for i in g.integers(0, len(pool), n)])
        return cu.render_shapes(attrs, 32, seed), np.array([CID[tuple(a)] for a in attrs])
    tr, trc = make(seen, 100_000, 1); te, tec = make(seen, 2000, 2); un, unc = make(unseen, 1200, 3)
    np.savez_compressed(f"{OUT}/shapes.npz", tr=tr, trc=trc, te=te, tec=tec, un=un, unc=unc)
    print(len(COMBOS), "captions;", len(seen), "seen;", len(WORDS), "words", WORDS)
    sys.exit()


class ImgEnc(nn.Module):
    def __init__(s, d=64):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 3, 2, 1), nn.GELU())
        s.p = nn.Linear(128 * 16, d)
    def forward(s, x): return s.p(s.c(x).flatten(1))


class TxtEnc(nn.Module):
    def __init__(s, vocab, d=64, w=128):
        super().__init__()
        s.e = nn.Embedding(vocab, w); s.pos = nn.Parameter(torch.zeros(8, w))
        s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, 4, 4 * w, 0.0, batch_first=True, norm_first=True), 2)
        s.p = nn.Linear(w, d)
    def forward(s, t): return s.p(s.t(s.e(t) + s.pos).mean(1))


def run(tag, loss_kind, B, tau, examples, seed, D):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    X = D["tr"]; Xc = D["trc"]
    ie, te = ImgEnc().to(dev), TxtEnc(len(WORDS)).to(dev)
    learned = tau == "learned"
    log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07) if learned else math.log(1 / float(tau)), device=dev),
                             requires_grad=learned)
    if loss_kind == "sigmoid":  # SigLIP init: t' = log 10, b = -10
        log_scale = nn.Parameter(torch.tensor(math.log(10.0), device=dev), requires_grad=learned)
        bias = nn.Parameter(torch.tensor(-10.0, device=dev))
    params = list(ie.parameters()) + list(te.parameters()) + [log_scale] + ([bias] if loss_kind == "sigmoid" else [])
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=2e-3, weight_decay=0.01)
    steps = examples // B
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, min(100, steps // 10))) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    tok = TOK.to(dev)
    mean = torch.tensor([0.25, 0.25, 0.25], device=dev).view(1, 3, 1, 1)
    def prep(x): return (torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255 - mean) / 0.25
    g = np.random.default_rng(seed)
    curve = []; acc_curve = []; t0 = time.time()
    for s in range(steps):
        idx = g.integers(0, len(X), B)
        c = torch.from_numpy(Xc[idx]).to(dev)
        img = F.normalize(ie(prep(X[idx])), dim=-1); txt = F.normalize(te(tok[c]), dim=-1)
        dup = (c[:, None] == c[None, :]) & ~torch.eye(B, dtype=torch.bool, device=dev)  # same caption, other item
        scale = log_scale.exp().clamp(max=100)
        logits = scale * img @ txt.t()
        if loss_kind == "softmax":
            logits = logits.masked_fill(dup, float("-inf"))
            lab = torch.arange(B, device=dev)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab)) / 2
        else:
            z = logits + bias
            sign = 2 * torch.eye(B, device=dev) - 1
            loss = -(F.logsigmoid(sign * z) * (~dup)).sum() / B
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if s % max(1, steps // 40) == 0 or s == steps - 1:
            curve.append((s * B, float(loss), float(scale)))
        if (s + 1) % max(1, steps // 8) == 0 and s + 1 < steps:
            with torch.no_grad():
                ie.eval(); te.eval()
                Tc = F.normalize(te(tok), dim=-1); Ic = F.normalize(ie(prep(D["te"][:500])), dim=-1)
                acc_curve.append(((s + 1) * B, float(((Ic @ Tc.t()).argmax(1).cpu().numpy() == D["tec"][:500]).mean())))
                ie.train(); te.train()
    torch.cuda.synchronize(); wall = time.time() - t0
    ie.eval(); te.eval()
    with torch.no_grad():
        T = F.normalize(te(tok), dim=-1)                                  # all 288 captions
        out = {}
        for split in ("te", "un"):
            I = F.normalize(ie(prep(D[split])), dim=-1)
            pred = (I @ T.t()).argmax(1).cpu().numpy(); truth = D[split + "c"]
            P = np.array([COMBOS[i] for i in pred]); Tr = np.array([COMBOS[i] for i in truth])
            out[split] = {"top1": float((pred == truth).mean()), "n": len(truth),
                          "attr": [float((P[:, a] == Tr[:, a]).mean()) for a in range(4)]}
        # eval loss at a fixed batch of 256 seen-test images (softmax CLIP loss, learned/used scale)
        I = F.normalize(ie(prep(D["te"][:256])), dim=-1); c = torch.from_numpy(D["tec"][:256]).to(dev)
        Tt = T[c]; dup = (c[:, None] == c[None, :]) & ~torch.eye(256, dtype=torch.bool, device=dev)
        lg = (log_scale.exp().clamp(max=100) * I @ Tt.t()).masked_fill(dup, float("-inf")); lab = torch.arange(256, device=dev)
        Ia = F.normalize(ie(prep(D["te"])), dim=-1); Ta = T[torch.from_numpy(D["tec"]).to(dev)]
        out["gap"] = float((Ia.mean(0) - Ta.mean(0)).norm()); out["cos_match"] = float((Ia * Ta).sum(1).mean())
        out["eval_loss_b256"] = float((F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab)) / 2)
    r = {"tag": tag, "loss": loss_kind, "B": B, "tau": tau, "examples": examples, "steps": steps, "seed": seed,
         "wall_s": wall, "final_scale": float(log_scale.exp().clamp(max=100)), "curve": curve, "acc_curve": acc_curve, **out}
    print(tag, loss_kind, B, tau, seed, "steps", steps, "wall %.1f" % wall, "seen %.3f unseen %.3f" % (out["te"]["top1"], out["un"]["top1"]),
          "scale %.1f" % r["final_scale"], "evloss %.3f" % out["eval_loss_b256"], "gap %.3f" % out["gap"], flush=True)
    with open(f"{OUT}/exp3-runs.jsonl", "a") as f: f.write(json.dumps(r) + "\n")


if sys.argv[1] == "run":
    D = dict(np.load(f"{OUT}/shapes.npz"))
    args = sys.argv[2:]
    for i in range(0, len(args), 6):
        tag, lk, B, tau, ex, seed = args[i:i + 6]
        run(tag, lk, int(B), tau, int(ex), int(seed), D)
