"""Shared helpers for Beyond Text Part 2 (vision transformers).

Loading pretrained classifiers, a legally usable labelled set mapped to ImageNet classes
(Oxford-IIIT Pet test split, CC BY-SA 4.0), resolution-controlled preprocessing, CUDA timing
with nvidia-smi sampling, mean attention distance, and the two small from-scratch models
(a tiny ViT and a small ResNet) used for the data-hunger run.
"""
import io, os, subprocess, threading, time, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HF_HOME = os.environ.get("HF_HOME", "/home/amit/.cache/vidyalai-scratch/hf")
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"

# Oxford-IIIT Pet breeds that have an exact ImageNet-1k class (index into the 1,000-way head).
# Egyptian Mau is left out: ImageNet's "Egyptian cat" is not that breed.
PETS_TO_IMAGENET = {
    "persian": 283, "siamese": 284,
    "american_pit_bull_terrier": 180, "basset_hound": 161, "beagle": 162, "boxer": 242,
    "chihuahua": 151, "english_cocker_spaniel": 219, "english_setter": 212,
    "german_shorthaired": 210, "great_pyrenees": 257, "japanese_chin": 152, "keeshond": 261,
    "leonberger": 255, "miniature_pinscher": 237, "newfoundland": 256, "pomeranian": 259,
    "pug": 254, "saint_bernard": 247, "samoyed": 258, "scottish_terrier": 199,
    "staffordshire_bull_terrier": 179, "wheaten_terrier": 202, "yorkshire_terrier": 187,
}

MEAN_STD = {
    "vit": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def pets_parquet(split="test"):
    pats = glob.glob(f"{HF_HOME}/hub/datasets--timm--oxford-iiit-pet/snapshots/*/data/{split}-*.parquet")
    assert pats, "Oxford-IIIT Pet parquet not downloaded"
    return pats[0]


def load_pets_mapped(split="test", limit=None):
    """Return (list of PIL RGB images, np.array of ImageNet labels, list of breed names)."""
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(pets_parquet(split))
    # timm/oxford-iiit-pet stores the breed as an integer label in alphabetical order (see the dataset card).
    imgs, labels, breeds = [], [], []
    classes = pets_class_names()
    for _, row in df.iterrows():
        breed = classes[int(row["label"])]
        if breed not in PETS_TO_IMAGENET:
            continue
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        imgs.append(im); labels.append(PETS_TO_IMAGENET[breed]); breeds.append(breed)
        if limit and len(imgs) >= limit:
            break
    return imgs, np.array(labels), breeds


def pets_class_names():
    """The 37 breed names in label order (alphabetical, lower-case, as in the dataset card)."""
    return sorted([
        "abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
        "bengal", "birman", "bombay", "boxer", "british_shorthair", "chihuahua", "egyptian_mau",
        "english_cocker_spaniel", "english_setter", "german_shorthaired", "great_pyrenees",
        "havanese", "japanese_chin", "keeshond", "leonberger", "maine_coon", "miniature_pinscher",
        "newfoundland", "persian", "pomeranian", "pug", "ragdoll", "russian_blue", "saint_bernard",
        "samoyed", "scottish_terrier", "shiba_inu", "siamese", "sphynx", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier"])


def to_tensor_batch(imgs, res, norm="vit"):
    """Resize the shorter side to `res` (bicubic, antialiased), centre-crop res x res, normalise."""
    import torchvision.transforms.v2.functional as TF
    mean, std = MEAN_STD[norm]
    out = []
    for im in imgs:
        t = TF.pil_to_tensor(im)
        t = TF.resize(t, res, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        t = TF.center_crop(t, [res, res])
        out.append(t)
    x = torch.stack(out).float().div_(255)
    m = torch.tensor(mean).view(1, 3, 1, 1); s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def load_hf(name, dtype=torch.float16, attn="sdpa"):
    from transformers import AutoModelForImageClassification, AutoModel
    kw = dict(dtype=dtype, device_map="cuda")
    if "dinov2" in name:
        return AutoModel.from_pretrained(name, attn_implementation=attn, **kw).eval()
    if "resnet" in name:
        return AutoModelForImageClassification.from_pretrained(name, **kw).eval()
    return AutoModelForImageClassification.from_pretrained(name, attn_implementation=attn, **kw).eval()


class GpuSampler:
    """Polls nvidia-smi for SM clock (MHz) and power (W) in a background thread."""
    def __init__(self, period=0.2):
        self.period, self.samples, self._stop = period, [], threading.Event()
    def _run(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run([NVSMI, "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                c, p, t = [float(v) for v in o.strip().split(",")]
                self.samples.append((c, p, t))
            except Exception:
                pass
            time.sleep(self.period)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join()
    def summary(self):
        if not self.samples:
            return {}
        a = np.array(self.samples)
        return {"sm_mhz_median": float(np.median(a[:, 0])), "power_w_median": float(np.median(a[:, 1])),
                "temp_c_max": float(a[:, 2].max()), "n_samples": len(a)}


def time_cuda(fn, warmup=3, reps=7):
    """Median-friendly list of wall times in ms, with torch.cuda.synchronize around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def patch_centers(gh, gw, patch):
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return np.stack([(ys.ravel() + 0.5) * patch, (xs.ravel() + 0.5) * patch], 1)


def mean_attention_distance(attn, gh, gw, patch, n_prefix=1):
    """attn: (B, H, N, N) attention probs incl. n_prefix non-patch tokens first.
    Returns (H,) mean distance in pixels: for each patch query, the attention-weighted mean
    distance to patch keys (weights renormalised over patch keys), averaged over queries and batch."""
    a = attn[:, :, n_prefix:, n_prefix:].float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    c = torch.tensor(patch_centers(gh, gw, patch), dtype=torch.float32, device=a.device)
    d = torch.cdist(c, c)  # (P, P) pixel distances
    return (a * d).sum(-1).mean(dim=(0, 2))  # (H,)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# ---------- small from-scratch models for CIFAR-10 (32 x 32) ----------

class TinyViT(nn.Module):
    """Pre-norm ViT: 4x4 patches (64 tokens) + class token, learned 1D positions."""
    def __init__(self, dim=256, depth=6, heads=4, mlp=512, patch=4, img=32, n_cls=10, pool="cls"):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, mlp, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, n_cls); self.pool = pool
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, 64, dim)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0] if self.pool == "cls" else x[:, 1:].mean(1))


def _conv_bn(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__(); self.a = _conv_bn(c, c); self.b = _conv_bn(c, c)
    def forward(self, x):
        return x + self.b(self.a(x))


class SmallResNet(nn.Module):
    """ResNet-9-style CNN (DAWNBench layout) at reduced width."""
    def __init__(self, w=(44, 88, 176, 352), n_cls=10):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn(3, w[0]), _conv_bn(w[0], w[1]), nn.MaxPool2d(2), _Res(w[1]),
            _conv_bn(w[1], w[2]), nn.MaxPool2d(2),
            _conv_bn(w[2], w[3]), nn.MaxPool2d(2), _Res(w[3]),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(w[3], n_cls))
    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cifar10_arrays():
    """CIFAR-10 as uint8 numpy arrays (N, 32, 32, 3) and int64 labels, from the HF parquet cache."""
    import pandas as pd
    from PIL import Image
    out = {}
    for split in ("train", "test"):
        p = glob.glob(f"{HF_HOME}/hub/datasets--uoft-cs--cifar10/snapshots/*/plain_text/{split}-*.parquet")[0]
        df = pd.read_parquet(p)
        x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
        out[split] = (x, df["label"].to_numpy().astype(np.int64))
    return out
