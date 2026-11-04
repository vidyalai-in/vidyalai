# Exp 3: tiny ViT vs small ResNet trained from scratch on CIFAR-10 subsets, same step budget.
# Usage: python train.py <steps> <cfg> [<cfg> ...]   cfg = model:n_train:seed   (results appended to results/train.jsonl)
import sys, json, time, math, os, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vit as L
OUT = "/home/amit/.cache/vidyalai-scratch/v02/results/train.jsonl"
steps = int(sys.argv[1]); cfgs = sys.argv[2:]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d["model"], d["n_train"], d["seed"], d["steps"]))
data = L.cifar10_arrays()
dev = "cuda"
Xtr = torch.from_numpy(data["train"][0]).permute(0, 3, 1, 2).contiguous().to(dev); Ytr = torch.from_numpy(data["train"][1]).to(dev)
Xte = torch.from_numpy(data["test"][0]).permute(0, 3, 1, 2).contiguous().to(dev); Yte = torch.from_numpy(data["test"][1]).to(dev)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
def norm(x): return ((x.float() / 255) - MEAN) / STD
def augment(x):  # random crop (pad 4, reflect) + horizontal flip, on GPU
    B = len(x); xp = F.pad(x.float(), (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=dev); j = torch.randint(0, 9, (B,), device=dev)
    idx = torch.arange(32, device=dev)
    rows = (i[:, None] + idx[None]); cols = (j[:, None] + idx[None])
    xp = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=dev) < 0.5
    xp[flip] = xp[flip].flip(-1)
    return xp
@torch.no_grad()
def evaluate(m):
    m.eval(); c = 0
    for s in range(0, len(Xte), 1000):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c += (m(norm(Xte[s:s + 1000])).argmax(-1) == Yte[s:s + 1000]).sum().item()
    m.train(); return c / len(Xte)
for cfg in cfgs:
    mname, n, seed = cfg.split(":"); n, seed = int(n), int(seed)
    if (mname, n, seed, steps) in done:
        print("skip", cfg); continue
    g = torch.Generator().manual_seed(1000 + seed)
    sub = torch.randperm(50000, generator=g)[:n].to(dev)   # same subset for both models at a given seed
    torch.manual_seed(seed)
    m = (L.TinyViT() if mname == "vit" else L.SmallResNet()).to(dev).to(memory_format=torch.channels_last)
    lr = 1e-3 if mname == "vit" else 2e-3
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.999))
    warm = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    B = 256; curve = []; t0 = time.time(); torch.cuda.synchronize()
    for step in range(steps):
        bi = sub[torch.randint(0, n, (B,), device=dev)]
        x = norm(augment(Xtr[bi])).contiguous(memory_format=torch.channels_last); y = Ytr[bi]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x), y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % (steps // 6) == 0 or step + 1 == steps:
            curve.append({"step": step + 1, "loss": float(loss), "test_acc": evaluate(m)})
    torch.cuda.synchronize(); wall = time.time() - t0
    # train accuracy on (up to 5,000 of) the subset, no augmentation
    with torch.no_grad():
        m.eval(); ii = sub[:5000]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tr = (m(norm(Xtr[ii])).argmax(-1) == Ytr[ii]).float().mean().item()
    rec = {"model": mname, "n_train": n, "seed": seed, "steps": steps, "batch": B, "lr": lr, "params": L.n_params(m),
           "wall_s": wall, "final_test_acc": curve[-1]["test_acc"], "train_acc": tr, "curve": curve}
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)
