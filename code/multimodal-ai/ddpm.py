"""DDPM / DDIM from scratch for Beyond Text Part 5 ("Diffusion from scratch").

Noise schedules (linear, Ho et al. 2020; cosine, Nichol and Dhariwal 2021), the closed-form forward process
q(x_t | x_0), an MLP denoiser for 2D toy data, a small U-Net with a sinusoidal time embedding for 28x28 images,
and two samplers that share one respaced timestep grid: DDPM ancestral steps and deterministic DDIM (eta = 0).

Usage:
    import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
    import ddpm
    s = ddpm.Schedule("linear")                    # or "cosine"; T = 1000
    xt = s.q_sample(x0, t, noise)                  # closed-form forward process
    x = ddpm.sample(model, (n, 1, 28, 28), s, steps=50, method="ddim")
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- schedules

def linear_alpha_bar(T=1000, beta_1=1e-4, beta_T=0.02):
    betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
    return torch.cumprod(1 - betas, 0)


def cosine_alpha_bar(T=1000, s=0.008, max_beta=0.999):
    f = lambda t: math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = torch.tensor([min(1 - f(t + 1) / f(t), max_beta) for t in range(T)], dtype=torch.float64)
    return torch.cumprod(1 - betas, 0)


class Schedule:
    """alpha_bar[t] for t = 0..T-1 (t = 0 is the first noise level, the paper's t = 1)."""

    def __init__(self, kind="linear", T=1000):
        self.kind, self.T = kind, T
        self.alpha_bar = (linear_alpha_bar if kind == "linear" else cosine_alpha_bar)(T)
        self.snr = self.alpha_bar / (1 - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar.to(x0.device)[t].float().view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


# ---------------------------------------------------------------- time embedding

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


# ---------------------------------------------------------------- 2D toy denoiser

class MLPDenoiser(nn.Module):
    def __init__(self, width=256, depth=4, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.inp = nn.Linear(2 + tdim, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, 2)

    def forward(self, x, t):
        h = F.silu(self.inp(torch.cat([x, timestep_embedding(t, self.tdim)], -1)))
        for l in self.hidden:
            h = F.silu(l(h))
        return self.out(h)


# ---------------------------------------------------------------- U-Net for 28x28

class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]          # the time embedding enters every block
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads, self.n = heads, nn.GroupNorm(8, ch)
        self.qkv, self.proj = nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(o)


class UNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28 with skip connections; self-attention at 7x7; predicts the noise."""

    def __init__(self, ch=64, mult=(1, 2, 2), nres=2, dropout=0.1, in_ch=1):
        super().__init__()
        tdim = 4 * ch
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down, chans, c = nn.ModuleList(), [ch], ch
        for i, m in enumerate(mult):
            for _ in range(nres):
                self.down.append(ResBlock(c, ch * m, tdim, dropout)); c = ch * m; chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim, dropout), Attention(c), ResBlock(c, c, tdim, dropout)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(nres + 1):
                self.up.append(ResBlock(c + chans.pop(), ch * m, tdim, dropout)); c = ch * m
                if i == len(mult) - 1:
                    self.up.append(Attention(c))
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.inp(x); hs = [h]
        for l in self.down:
            h = l(h, temb) if isinstance(l, ResBlock) else l(h)
            hs.append(h)
        for l in self.mid:
            h = l(h, temb) if isinstance(l, ResBlock) else l(h)
        for l in self.up:
            if isinstance(l, ResBlock):
                h = l(torch.cat([h, hs.pop()], 1), temb)
            else:
                h = l(h)
        return self.out(h)


# ---------------------------------------------------------------- training loss

def loss_simple(model, x0, sched):
    """Ho et al.'s L_simple: mean-squared error between the true and the predicted noise at a uniform random t."""
    t = torch.randint(0, sched.T, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    return F.mse_loss(model(sched.q_sample(x0, t, noise), t), noise)


# ---------------------------------------------------------------- samplers

def timesteps(T, steps):
    """Evenly spaced subset of the T training noise levels, from the noisiest down to t = 0."""
    return torch.linspace(T - 1, 0, steps).round().long().tolist()


@torch.no_grad()
def sample(model, shape, sched, steps=1000, method="ddim", generator=None, device="cuda", clip=True,
           return_traj=False, dtype=torch.bfloat16):
    """DDPM ancestral sampling (posterior variance, i.e. DDIM eta = 1) or DDIM (eta = 0) on the same grid.

    With steps < T the chain is respaced: each jump t -> s uses alpha_bar[t] and alpha_bar[s] directly,
    so the per-jump beta is 1 - alpha_bar[t] / alpha_bar[s] (Nichol and Dhariwal 2021, section 4).
    """
    ab = sched.alpha_bar.to(device)
    x = torch.randn(shape, generator=generator, device=device)
    ts = timesteps(sched.T, steps)
    traj = [x.float().cpu()] if return_traj else None
    for i, t in enumerate(ts):
        a_t = ab[t]
        a_s = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, dtype=ab.dtype, device=device)
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            eps = model(x, tt).float()
        x0 = (x - (1 - a_t).sqrt().float() * eps) / a_t.sqrt().float()
        if clip:
            x0 = x0.clamp(-1, 1)
        eps = (x - a_t.sqrt().float() * x0) / (1 - a_t).sqrt().float()   # keep eps consistent with clipped x0
        if method == "ddim":
            x = a_s.sqrt().float() * x0 + (1 - a_s).sqrt().float() * eps
        else:
            beta = 1 - a_t / a_s
            var = (1 - a_s) / (1 - a_t) * beta                              # posterior variance (tilde beta)
            mean = (a_s.sqrt() * beta / (1 - a_t)).float() * x0 + ((1 - beta).sqrt() * (1 - a_s) / (1 - a_t)).float() * x
            x = mean
            if i + 1 < len(ts):
                x = x + var.sqrt().float() * torch.randn(shape, generator=generator, device=device)
        if return_traj:
            traj.append(x.float().cpu())
    return (x, traj) if return_traj else x


class EMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)
