"""Experiment 1b: autoregressive (raster order) and MaskGIT-style priors over VQ-VAE codes on Fashion-MNIST.
Usage:
  python prior_run.py codes <vq ckpt> <K> <down>     encode train/test into codes (codes.pt)
  python prior_run.py train <mode ar|mask> <steps_this_chunk> <total_steps>   resumable chunks
  python prior_run.py sample                          FD-FMNIST for each sampler setting, 5,000 x 3 seeds
  python prior_run.py time                            samples/s and latency
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import torch

import ar_image as ai
import fmnist_eval as fe

torch.set_float32_matmul_precision("high")
V = Path("/home/amit/.cache/vidyalai-scratch/v08")
RUNS, CK = V / "runs", V / "ckpt"
CFG = json.loads((V / "prior_cfg.json").read_text()) if (V / "prior_cfg.json").exists() else {}


def load_vq():
    m = ai.VQVAE(K=CFG["K"], down=CFG["down"]).cuda()
    m.load_state_dict(torch.load(CFG["vq"], map_location="cuda"))
    return m.eval()


def cmd_codes(ck, K, down):
    cfg = {"vq": ck, "K": int(K), "down": int(down)}
    (V / "prior_cfg.json").write_text(json.dumps(cfg))
    CFG.update(cfg)
    vq = load_vq()
    out = {}
    with torch.no_grad():
        for split in ("train", "test"):
            x, _ = fe.load(split)
            out[split] = torch.cat([vq.encode(x[i:i + 2000].cuda()).flatten(1).cpu() for i in range(0, len(x), 2000)])
    torch.save(out, CK / "codes.pt")
    print({k: tuple(v.shape) for k, v in out.items()})


def cmd_train(mode, chunk, total):
    chunk, total = int(chunk), int(total)
    codes = torch.load(CK / "codes.pt")
    N = codes["train"].shape[1]
    path = CK / f"prior_{mode}.pt"
    m = ai.Prior(CFG["K"], N, mode).cuda()
    opt = torch.optim.AdamW(m.parameters(), 6e-4, betas=(0.9, 0.95), weight_decay=0.05)
    start = 0
    if path.exists():
        st = torch.load(path, map_location="cuda")
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
    steps = min(chunk, total - start)
    if steps <= 0:
        print("already done", start); return
    t0 = time.time()
    m, opt, log = ai.train_prior(codes["train"], CFG["K"], mode, steps=steps, model=m, opt=opt,
                                 start_step=start, total_steps=total)
    torch.cuda.synchronize()
    sec = time.time() - t0
    nll = ai.prior_nll(m, codes["test"])
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": start + steps}, path)
    with open(RUNS / f"prior_{mode}_train.jsonl", "a") as f:
        f.write(json.dumps({"start": start, "end": start + steps, "sec": sec, "test_nll": nll, "curve": log}) + "\n")
    print(mode, start + steps, round(sec, 1), "test nll", round(nll, 4),
          "params", sum(p.numel() for p in m.parameters()), flush=True)


def load_prior(mode):
    codes = torch.load(CK / "codes.pt")
    N = codes["train"].shape[1]
    m = ai.Prior(CFG["K"], N, mode).cuda()
    m.load_state_dict(torch.load(CK / f"prior_{mode}.pt", map_location="cuda")["model"])
    return m.eval(), N


@torch.no_grad()
def decode(vq, c, side):
    return torch.cat([vq.decode(c[i:i + 1000].view(-1, side, side)).clamp(-1, 1).cpu() for i in range(0, len(c), 1000)])


def gen(fn, n=5000, bs=1000):
    return torch.cat([fn(bs) for _ in range(n // bs)])


def cmd_sample(settings=None):
    vq = load_vq()
    ar, N = load_prior("ar")
    mk, _ = load_prior("mask")
    side = int(N ** 0.5)
    out = RUNS / "prior_fd.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(l)["name"], json.loads(l)["seed"]) for l in out.read_text().splitlines()}
    sets = [("ar_t1.0", lambda b: ai.ar_sample(ar, b, 1.0)),
            ("ar_t0.9", lambda b: ai.ar_sample(ar, b, 0.9))]
    for T in (1, 2, 4, 8, 12, 16, N):
        sets.append((f"mask_T{T}", lambda b, T=T: ai.mask_sample(mk, b, T=T)))
    for name, fn in sets:
        for seed in (0, 1, 2):
            if (name, seed) in done:
                continue
            torch.manual_seed(seed)
            c = gen(fn)
            x = decode(vq, c, side)
            r = fe.score(x)
            r.update({"name": name, "seed": seed})
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            if seed == 0:
                torch.save(x[:64].half(), RUNS / f"grid_{name}.pt")
            print(name, seed, round(r["fd"], 2), flush=True)
    # reference: the tokenizer's own reconstructions of 5,000 test-distribution-free training images
    x, _ = fe.load("train")
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(1))[:5000]
    codes = torch.load(CK / "codes.pt")["train"][idx]
    r = fe.score(decode(vq, codes.cuda(), side))
    r.update({"name": "vq_recon_train5000", "seed": 1})
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print("recon", r["fd"])


def cmd_time():
    vq = load_vq()
    ar, N = load_prior("ar")
    mk, _ = load_prior("mask")
    side = int(N ** 0.5)
    res = {}
    for B in (1, 500):
        res[f"ar_b{B}"] = ai.time_cuda(lambda: ai.ar_sample(ar, B), runs=7, warmup=3)
        for T in (4, 8, 16):
            res[f"mask_T{T}_b{B}"] = ai.time_cuda(lambda: ai.mask_sample(mk, B, T=T), runs=7, warmup=3)
        c = torch.randint(0, CFG["K"], (B, side, side), device="cuda")
        with torch.no_grad():
            res[f"decode_b{B}"] = ai.time_cuda(lambda: vq.decode(c), runs=9, warmup=3)
        # AR without the KV cache: recompute the whole prefix every step
        def ar_nocache():
            t = torch.full((B, 1), ar.special, device="cuda")
            for i in range(N):
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    lg = ar(t)[:, -1].float()
                t = torch.cat([t, torch.multinomial(torch.softmax(lg, -1), 1)], 1)
        res[f"ar_nocache_b{B}"] = ai.time_cuda(ar_nocache, runs=5, warmup=2)
        print(B, {k: round(v["median_s"] * 1000, 2) for k, v in res.items() if k.endswith(f"b{B}")}, flush=True)
    res["params_ar"] = sum(p.numel() for p in ar.parameters())
    res["params_mask"] = sum(p.numel() for p in mk.parameters())
    res["params_vq"] = sum(p.numel() for p in vq.parameters())
    res["params_vq_decoder"] = sum(p.numel() for p in vq.dec.parameters()) + vq.codebook.weight.numel()
    (RUNS / "prior_timing.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    {"codes": cmd_codes, "train": cmd_train, "sample": cmd_sample, "time": cmd_time}[c](*sys.argv[2:])
