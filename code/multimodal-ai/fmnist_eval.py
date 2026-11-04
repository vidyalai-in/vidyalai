"""Shared Fashion-MNIST data and sample-quality metric for Beyond Text Parts 5-8.

Fashion-MNIST (Xiao et al., 2017, arXiv 1708.07747; MIT license, github.com/zalandoresearch/fashion-mnist):
60,000 training and 10,000 test grayscale 28x28 images of clothing in 10 classes.

The metric is an FID-style Frechet distance computed on the 128-d penultimate features of a small CNN
classifier trained here on the Fashion-MNIST training split ("FD-FMNIST"), not on Inception-v3 features,
so its values are NOT comparable with published FID numbers. Use it to compare our own models with each other.

Usage:
    import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
    import fmnist_eval as fe
    x, y = fe.load("train")          # float32 tensors: x in [-1, 1], shape (N, 1, 28, 28); y int64
    r = fe.score(samples)            # samples: tensor (N, 1, 28, 28) in [-1, 1], N >= 2,000 recommended
    # r = {"fd": ..., "class_hist": [...], "mean_conf": ..., "n": ...}

Reference values (printed by `python fmnist_eval.py --reference`) are stored in REFERENCE below.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/amit/.cache/vidyalai-scratch/mm/data")
NPZ = ROOT / "fmnist.npz"
CLS = ROOT / "fmnist_cls.pt"
STATS = ROOT / "fmnist_test_stats.npz"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
REFERENCE = {}  # filled from fmnist_reference.json if present


def _cache():
    if NPZ.exists():
        return
    from datasets import load_dataset
    d = load_dataset("zalando-datasets/fashion_mnist")
    out = {}
    for split in ("train", "test"):
        out[f"{split}_x"] = np.stack([np.asarray(im, dtype=np.uint8) for im in d[split]["image"]])
        out[f"{split}_y"] = np.asarray(d[split]["label"], dtype=np.int64)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **out)


def load(split="train"):
    """Return (x, y): x float32 in [-1, 1], shape (N, 1, 28, 28); y int64 labels."""
    _cache()
    z = np.load(NPZ)
    x = torch.from_numpy(z[f"{split}_x"]).float().div(127.5).sub(1).unsqueeze(1)
    return x, torch.from_numpy(z[f"{split}_y"])


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 3 * 3, 128)
        self.out = nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 7
        x = F.max_pool2d(F.relu(self.c3(x)), 2)   # 3
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        return self.out(self.features(x))


_model = None


def classifier(device=None):
    global _model
    if _model is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = Classifier()
        m.load_state_dict(torch.load(CLS, map_location="cpu"))
        _model = m.to(device).eval()
    return _model


@torch.no_grad()
def features(x, batch=1000):
    """128-d penultimate features and softmax probabilities for images in [-1, 1]."""
    m = classifier()
    dev = next(m.parameters()).device
    fs, ps = [], []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev, torch.float32).clamp(-1, 1)
        f = m.features(xb)
        fs.append(f.cpu())
        ps.append(F.softmax(m.out(f), -1).cpu())
    return torch.cat(fs).double().numpy(), torch.cat(ps).numpy()


def _frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    eps = 1e-6 * np.eye(len(s1))  # dead ReLU units give zero-variance features; keep the product well conditioned
    s1, s2 = s1 + eps, s2 + eps
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def _stats(f):
    return f.mean(0), np.cov(f, rowvar=False)


def test_stats():
    if not STATS.exists():
        x, _ = load("test")
        f, _ = features(x)
        mu, s = _stats(f)
        np.savez(STATS, mu=mu, s=s)
    z = np.load(STATS)
    return z["mu"], z["s"]


def score(samples):
    """FD-FMNIST against the 10,000 test images, class histogram and mean top-class confidence."""
    f, p = features(samples)
    mu, s = _stats(f)
    mu_t, s_t = test_stats()
    hist = np.bincount(p.argmax(1), minlength=10) / len(p)
    return {"fd": _frechet(mu, s, mu_t, s_t), "class_hist": hist.round(4).tolist(),
            "mean_conf": float(p.max(1).mean()), "n": int(len(samples))}


def train_classifier(epochs=8, seed=0):
    torch.manual_seed(seed)
    x, y = load("train")
    xt, yt = load("test")
    m = Classifier().cuda()
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(x) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    xg, yg = x.cuda(), y.cuda()
    for ep in range(epochs):
        perm = torch.randperm(len(x), device="cuda")
        for i in range(0, len(x) - 255, 256):
            idx = perm[i:i + 256]
            xb = xg[idx]
            # light augmentation: random 1-px shifts and horizontal flips
            if torch.rand(()) < 0.5:
                xb = xb.flip(-1)
            xb = torch.roll(xb, shifts=(int(torch.randint(-1, 2, ())), int(torch.randint(-1, 2, ()))), dims=(2, 3))
            loss = F.cross_entropy(m(xb), yg[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        acc = (m(xt.cuda()).argmax(1).cpu() == yt).float().mean().item()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(m.cpu().state_dict(), CLS)
    return acc


if __name__ == "__main__":
    if "--train" in sys.argv:
        print("test accuracy", train_classifier())
    if "--reference" in sys.argv:
        STATS.unlink(missing_ok=True)
        x, y = load("train")
        g = torch.Generator().manual_seed(0)
        ref = {}
        for n in (2000, 5000, 10000):
            idx = torch.randperm(len(x), generator=g)[:n]
            ref[f"train_{n}_vs_test"] = score(x[idx])["fd"]
        noise = torch.rand(5000, 1, 28, 28) * 2 - 1
        ref["uniform_noise_5000"] = score(noise)["fd"]
        blur = F.avg_pool2d(x[:5000], 3, 1, 1)
        ref["train_5000_blurred3x3"] = score(blur)["fd"]
        _, yt = load("test")
        xt, _ = load("test")
        _, p = features(xt)
        ref["classifier_test_acc"] = float((p.argmax(1) == yt.numpy()).mean())
        (ROOT / "fmnist_reference.json").write_text(json.dumps(ref, indent=1))
        print(json.dumps(ref, indent=1))

_ref = ROOT / "fmnist_reference.json"
if _ref.exists():
    REFERENCE.update(json.loads(_ref.read_text()))
