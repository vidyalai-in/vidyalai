"""Shared code for Beyond Text Part 8 (autoregressive and masked image generation) on Fashion-MNIST.

- VQVAE: a small convolutional VQ-VAE (van den Oord et al. 2017, arXiv 1711.00937) that turns a 28x28 image
  into a grid of integer codes (7x7 with down=4, 14x14 with down=2). Plain MSE reconstruction loss, codebook loss
  and commitment loss (beta = 0.25), straight-through gradients. Optional dead-code restart: every `restart_every`
  steps, codes unused in that window are moved onto random encoder outputs from the current batch.
- Prior: one transformer used two ways over the flattened code grid (raster order):
    mode="ar"   causal attention, input [BOS, c0..c_{N-2}], predicts c0..c_{N-1} (DALL-E / Parti / LlamaGen style)
    mode="mask" bidirectional attention, a cosine-scheduled share of tokens replaced by [MASK], loss on masked
                positions only (MaskGIT, arXiv 2202.04200)
- Samplers: ar_sample (with a KV cache, one forward pass per token) and mask_sample (MaskGIT iterative decoding:
  T forward passes, cosine schedule, confidence = log-prob of the sampled token plus annealed Gumbel noise).
- gpu_monitor: samples SM clock and power with nvidia-smi in a thread while a timed block runs.
"""
import math
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NVSMI = "/usr/lib/wsl/lib/nvidia-smi"


# ----------------------------------------------------------------------------- VQ-VAE
class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.b(x)


class VQVAE(nn.Module):
    def __init__(self, K=512, d=64, ch=128, down=4, beta=0.25):
        super().__init__()
        self.K, self.d, self.beta, self.down = K, d, beta, down
        enc = [nn.Conv2d(1, ch // 2, 4, 2, 1), nn.ReLU()]
        if down == 4:
            enc += [nn.Conv2d(ch // 2, ch, 4, 2, 1)]
        else:
            enc += [nn.Conv2d(ch // 2, ch, 3, 1, 1)]
        enc += [Res(ch), Res(ch), nn.ReLU(), nn.Conv2d(ch, d, 1)]
        self.enc = nn.Sequential(*enc)
        dec = [nn.Conv2d(d, ch, 3, padding=1), Res(ch), Res(ch), nn.ReLU()]
        if down == 4:
            dec += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(ch // 2, 1, 4, 2, 1)]
        else:
            dec += [nn.Conv2d(ch, ch // 2, 3, 1, 1), nn.ReLU(), nn.ConvTranspose2d(ch // 2, 1, 4, 2, 1)]
        self.dec = nn.Sequential(*dec)
        self.codebook = nn.Embedding(K, d)
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def quantize(self, z):
        # z: (B, d, H, W) -> nearest codebook entry per cell
        B, d, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, d)
        e = self.codebook.weight
        dist = zf.pow(2).sum(1, keepdim=True) - 2 * zf @ e.t() + e.pow(2).sum(1)[None]
        idx = dist.argmin(1)
        zq = e[idx].view(B, H, W, d).permute(0, 3, 1, 2)
        return zq, idx.view(B, H, W), zf

    def encode(self, x):
        return self.quantize(self.enc(x))[1]

    def decode(self, idx):
        zq = self.codebook(idx).permute(0, 3, 1, 2)
        return self.dec(zq)

    def forward(self, x):
        z = self.enc(x)
        zq, idx, zf = self.quantize(z)
        codebook_loss = F.mse_loss(zq, z.detach())
        commit_loss = F.mse_loss(z, zq.detach())
        zq_st = z + (zq - z).detach()  # straight-through estimator
        xr = self.dec(zq_st)
        rec = F.mse_loss(xr, x)
        return xr, idx, rec, codebook_loss + self.beta * commit_loss, zf


def train_vqvae(x, K=512, down=4, steps=3000, bs=256, lr=2e-4, restart_every=0, seed=0, log_every=250):
    torch.manual_seed(seed)
    m = VQVAE(K=K, down=down).cuda()
    opt = torch.optim.Adam(m.parameters(), lr)
    xg = x.cuda()
    used = torch.zeros(K, dtype=torch.bool, device="cuda")
    log = []
    t0 = time.time()
    for s in range(1, steps + 1):
        xb = xg[torch.randint(0, len(xg), (bs,), device="cuda")]
        xr, idx, rec, vql, zf = m(xb)
        loss = rec + vql
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        used[idx.flatten()] = True
        if restart_every and s % restart_every == 0:
            dead = (~used).nonzero().flatten()
            if len(dead):
                pick = zf.detach()[torch.randint(0, len(zf), (len(dead),), device="cuda")]
                m.codebook.weight.data[dead] = pick
            used.zero_()
        if s % log_every == 0 or s == steps:
            log.append({"step": s, "rec": rec.item(), "vq": vql.item(), "sec": time.time() - t0})
    return m.eval(), log


@torch.no_grad()
def vq_eval(m, x, bs=2000):
    """Reconstruction MSE (images in [-1,1]), PSNR on the [0,1] scale, codebook usage and perplexity."""
    counts = torch.zeros(m.K, device="cuda")
    se, n = 0.0, 0
    recs = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].cuda()
        idx = m.encode(xb)
        xr = m.decode(idx).clamp(-1, 1)
        se += (xr - xb).pow(2).sum().item()
        n += xb.numel()
        counts += torch.bincount(idx.flatten(), minlength=m.K).float()
        recs.append(xr.cpu())
    mse = se / n
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum().item()
    return {"mse": mse, "psnr": 10 * math.log10(1.0 / (mse / 4)), "used": int((counts > 0).sum().item()),
            "used_frac": float((counts > 0).float().mean().item()), "perplexity": math.exp(ent)}, torch.cat(recs)


# ----------------------------------------------------------------------------- transformer prior
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Prior(nn.Module):
    def __init__(self, K, N, mode="ar", d=256, L=6, h=8):
        super().__init__()
        self.K, self.N, self.mode = K, N, mode
        self.special = K  # BOS for "ar", MASK for "mask"
        self.tok = nn.Embedding(K + 1, d)
        self.pos = nn.Parameter(torch.zeros(1, N, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, K)

    def forward(self, t, start=0, caches=None):
        x = self.tok(t) + self.pos[:, start:start + t.shape[1]]
        for i, b in enumerate(self.blocks):
            x = b(x, self.mode == "ar", None if caches is None else caches[i])
        return self.head(self.ln(x))

    def loss(self, c):
        B, N = c.shape
        if self.mode == "ar":
            inp = torch.cat([torch.full((B, 1), self.special, device=c.device), c[:, :-1]], 1)
            return F.cross_entropy(self(inp).reshape(-1, self.K), c.reshape(-1))
        r = torch.rand(B, 1, device=c.device)
        n_mask = torch.ceil(torch.cos(r * math.pi / 2) * N).clamp(min=1)
        rank = torch.rand(B, N, device=c.device).argsort(1).argsort(1)
        m = rank < n_mask
        inp = torch.where(m, torch.full_like(c, self.special), c)
        logits = self(inp)
        return F.cross_entropy(logits[m], c[m])


def train_prior(codes, K, mode, steps=6000, bs=256, lr=6e-4, seed=0, d=256, L=6, log_every=250, model=None, opt=None, start_step=0, total_steps=None):
    torch.manual_seed(seed + start_step)
    N = codes.shape[1]
    m = model or Prior(K, N, mode, d=d, L=L).cuda()
    opt = opt or torch.optim.AdamW(m.parameters(), lr, betas=(0.9, 0.95), weight_decay=0.05)
    total = total_steps or steps
    cg = codes.cuda()
    log = []
    t0 = time.time()
    m.train()
    for s in range(start_step + 1, start_step + steps + 1):
        lr_s = lr * min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
        for g in opt.param_groups:
            g["lr"] = lr_s
        cb = cg[torch.randint(0, len(cg), (bs,), device="cuda")]
        with torch.autocast("cuda", torch.bfloat16):
            loss = m.loss(cb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % log_every == 0:
            log.append({"step": s, "loss": loss.item(), "sec": time.time() - t0})
    return m.eval(), opt, log


@torch.no_grad()
def prior_nll(m, codes, bs=1000):
    """Mean cross-entropy (nats/token) on held-out codes. For 'mask' mode it is averaged over random masks."""
    tot, n = 0.0, 0
    g = torch.Generator(device="cuda").manual_seed(123)
    for i in range(0, len(codes), bs):
        c = codes[i:i + bs].cuda()
        with torch.autocast("cuda", torch.bfloat16):
            if m.mode == "ar":
                l = m.loss(c)
            else:
                torch.manual_seed(1000 + i)
                l = m.loss(c)
        tot += l.item() * len(c)
        n += len(c)
    return tot / n


@torch.no_grad()
def ar_sample(m, B, temperature=1.0, top_k=0, generator=None):
    """Raster-order sampling with a KV cache: N forward passes, each on one new token."""
    caches = [dict() for _ in m.blocks]
    t = torch.full((B, 1), m.special, device="cuda")
    out = []
    for i in range(m.N):
        with torch.autocast("cuda", torch.bfloat16):
            logits = m(t, start=i, caches=caches)[:, -1].float() / temperature
        if top_k:
            v, _ = logits.topk(top_k, -1)
            logits[logits < v[:, -1:]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
        out.append(nxt)
        t = nxt
    return torch.cat(out, 1)


@torch.no_grad()
def mask_sample(m, B, T=8, temperature=1.0, choice_temp=4.5, generator=None):
    """MaskGIT decoding: T forward passes. After pass t, ceil(N * cos(pi/2 * t/T)) tokens stay masked;
    the ones kept are those with the highest log-prob of their sampled value plus Gumbel noise scaled by
    choice_temp * (1 - t/T)."""
    N = m.N
    c = torch.full((B, N), m.special, device="cuda")
    for t in range(1, T + 1):
        masked = c == m.special
        with torch.autocast("cuda", torch.bfloat16):
            logits = m(c).float() / temperature
        probs = F.softmax(logits, -1)
        samp = torch.multinomial(probs.view(-1, m.K), 1, generator=generator).view(B, N)
        samp = torch.where(masked, samp, c)
        logp = torch.log(probs.gather(-1, samp[..., None])[..., 0] + 1e-10)
        n_keep_masked = int(math.floor(N * math.cos(math.pi / 2 * t / T)))
        if n_keep_masked == 0:
            c = samp
            break
        u = torch.rand(B, N, device="cuda", generator=generator).clamp(1e-10, 1 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        conf = logp + choice_temp * (1 - t / T) * gumbel
        conf = torch.where(masked, conf, torch.full_like(conf, float("inf")))
        # the n_keep_masked lowest-confidence positions stay masked (never more than are masked now)
        k = min(n_keep_masked, int(masked.sum(1).min().item()) - 1) if t < T else 0
        k = max(k, 0)
        thresh = conf.sort(1).values[:, k - 1:k] if k > 0 else None
        c = samp
        if k > 0:
            c = torch.where(conf <= thresh, torch.full_like(c, m.special), c)
    return c


# ----------------------------------------------------------------------------- timing helpers
class gpu_monitor:
    """Context manager that samples SM clock (MHz) and power (W) every ~200 ms."""

    def __enter__(self):
        self.samples, self._stop = [], False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def _run(self):
        while not self._stop:
            try:
                o = subprocess.run([NVSMI, "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.samples.append((float(o[0]), float(o[1])))
            except Exception:
                pass
            time.sleep(0.2)

    def __exit__(self, *a):
        self._stop = True
        self.th.join()

    def summary(self):
        import statistics as st
        if not self.samples:
            return {}
        return {"sm_mhz": st.median(s[0] for s in self.samples), "power_w": st.median(s[1] for s in self.samples),
                "n_samples": len(self.samples)}


def time_cuda(fn, runs=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    with gpu_monitor() as mon:
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_s": ts[len(ts) // 2], "min_s": ts[0], "max_s": ts[-1], "runs": runs, **mon.summary()}
