"""Shared code for Beyond Text Part 7 (diffusion transformers and flow matching), writer v07.

Two denoiser networks for 28x28 single-channel images, both taking (x, t) with t a float in [0, 1]:
  UNet  - a small DDPM-style U-Net (ResBlocks + GroupNorm, time embedding added in every block,
          self-attention at 7x7), 28 -> 14 -> 7.
  DiT   - a small diffusion transformer (Peebles and Xie, arXiv 2212.09748): patchify, fixed 2D sin-cos
          position embedding, adaLN-Zero blocks, linear unpatchify.

Two objectives with the same networks:
  ddpm_loss - DDPM epsilon prediction (Ho et al., arXiv 2006.11239), linear beta schedule 1e-4..0.02, T=1000.
  flow_loss - rectified flow / flow matching with straight paths x_t = (1-t) noise + t data, target velocity
              data - noise (Liu et al. arXiv 2209.03003; Lipman et al. arXiv 2210.02747). t=0 is noise, t=1 data.
              t ~ uniform, or logit-normal (Esser et al. arXiv 2403.03206).

Samplers: ddim_sample (deterministic DDIM, eta = 0, arXiv 2010.02502) and euler_sample (Euler ODE steps).
Path diagnostics: path_stats(traj) -> length/chord ratio and Liu et al.'s straightness S.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- time embedding
def timestep_embedding(t, dim, scale=1000.0):
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000 like integer DDPM steps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * scale)[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class TimeMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, t):
        return self.net(timestep_embedding(t, self.dim))


# ---------------------------------------------------------------- U-Net
class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn2d(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)
        self.h = heads

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, hh * ww).transpose(-1, -2).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v)            # (b, heads, hw, c/heads)
        return x + self.o(y.transpose(-1, -2).reshape(b, c, hh, ww))


class UNet(nn.Module):
    def __init__(self, ch=(48, 96, 128), tdim=192):
        super().__init__()
        c0, c1, c2 = ch
        self.temb = TimeMLP(64, tdim)
        self.inp = nn.Conv2d(1, c0, 3, padding=1)
        self.d0 = nn.ModuleList([ResBlock(c0, c0, tdim), ResBlock(c0, c0, tdim)])
        self.down0 = nn.Conv2d(c0, c0, 3, stride=2, padding=1)       # 28 -> 14
        self.d1 = nn.ModuleList([ResBlock(c0, c1, tdim), ResBlock(c1, c1, tdim)])
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)       # 14 -> 7
        self.d2 = nn.ModuleList([ResBlock(c1, c2, tdim), ResBlock(c2, c2, tdim)])
        self.mid1, self.mida, self.mid2 = ResBlock(c2, c2, tdim), Attn2d(c2), ResBlock(c2, c2, tdim)
        self.u2 = nn.ModuleList([ResBlock(c2 + c2, c2, tdim), ResBlock(c2 + c2, c2, tdim)])
        self.up1 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)  # 7 -> 14
        self.u1 = nn.ModuleList([ResBlock(c2 + c1, c1, tdim), ResBlock(c1 + c1, c1, tdim)])
        self.up0 = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)  # 14 -> 28
        self.u0 = nn.ModuleList([ResBlock(c1 + c0, c0, tdim), ResBlock(c0 + c0, c0, tdim)])
        self.out = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        e = self.temb(t)
        h = self.inp(x)
        s = []
        for b in self.d0: h = b(h, e); s.append(h)
        h = self.down0(h)
        for b in self.d1: h = b(h, e); s.append(h)
        h = self.down1(h)
        for b in self.d2: h = b(h, e); s.append(h)
        h = self.mid2(self.mida(self.mid1(h, e)), e)
        for b in self.u2: h = b(torch.cat([h, s.pop()], 1), e)
        h = self.up1(h)
        for b in self.u1: h = b(torch.cat([h, s.pop()], 1), e)
        h = self.up0(h)
        for b in self.u0: h = b(torch.cat([h, s.pop()], 1), e)
        return self.out(h)


# ---------------------------------------------------------------- DiT
def sincos_2d(dim, n):
    """Fixed 2D sin-cos position embedding for an n x n grid, as in DiT/MAE. Returns (n*n, dim)."""
    def one(d, pos):
        om = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        a = pos.reshape(-1).double()[:, None] * om[None]
        return torch.cat([a.sin(), a.cos()], 1)
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return torch.cat([one(dim // 2, yy), one(dim // 2, xx)], 1).float()


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Pre-norm Transformer block whose LayerNorms are adaptive: shift, scale and a residual gate
    come from the conditioning vector. The gate's projection starts at zero (adaLN-Zero), so each
    block starts as the identity."""
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(approximate="tanh"), nn.Linear(mlp * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
        self.h = heads

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, -1)
        b, n, d = x.shape
        q, k, v = self.qkv(modulate(self.n1(x), sh1, sc1)).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + g1[:, None] * self.proj(a)
        x = x + g2[:, None] * self.mlp(modulate(self.n2(x), sh2, sc2))
        return x


class DiT(nn.Module):
    def __init__(self, size=28, patch=2, d=192, depth=6, heads=6):
        super().__init__()
        self.p, self.g = patch, size // patch
        self.embed = nn.Conv2d(1, d, patch, stride=patch)            # patchify: one strided conv
        self.register_buffer("pos", sincos_2d(d, self.g)[None], persistent=False)
        self.temb = TimeMLP(256, d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaf = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final = nn.Linear(d, patch * patch)
        for m in (self.adaf[-1], self.final):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        b = x.shape[0]
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos       # (b, g*g, d)
        c = self.temb(t)
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.adaf(c).chunk(2, -1)
        h = self.final(modulate(self.nf(h), sh, sc))                  # (b, g*g, p*p)
        h = h.reshape(b, self.g, self.g, self.p, self.p).permute(0, 1, 3, 2, 4)
        return h.reshape(b, 1, self.g * self.p, self.g * self.p)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- objectives
T_DDPM = 1000


def ddpm_alpha_bar(device="cpu"):
    betas = torch.linspace(1e-4, 0.02, T_DDPM, dtype=torch.float64)
    return torch.cumprod(1 - betas, 0).float().to(device)            # abar[i] for i = 0..999


def ddpm_loss(model, x0, abar):
    """Predict the added noise; t index i in 0..999 is fed to the net as i / 1000."""
    i = torch.randint(0, T_DDPM, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    a = abar[i][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return F.mse_loss(model(xt, i.float() / T_DDPM).float(), eps)


def sample_t(n, device, kind="uniform"):
    if kind == "uniform":
        return torch.rand(n, device=device)
    if kind == "logit_normal":                                        # Esser et al.: m = 0, s = 1
        return torch.sigmoid(torch.randn(n, device=device))
    raise ValueError(kind)


def flow_loss(model, x1, t_kind="uniform", x0=None):
    """Rectified flow: x_t = (1 - t) x0 + t x1 with x0 ~ N(0, I) (or a given paired noise), target x1 - x0."""
    x0 = torch.randn_like(x1) if x0 is None else x0
    t = sample_t(x1.shape[0], x1.device, t_kind)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1
    return F.mse_loss(model(xt, t).float(), x1 - x0)


# ---------------------------------------------------------------- samplers
def _amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def euler_sample(model, z, steps, keep_traj=False):
    """Integrate dx/dt = v(x, t) from t=0 (noise z) to t=1 with `steps` equal Euler steps."""
    x = z.clone()
    traj = [x.float().cpu()] if keep_traj else None
    ts = torch.linspace(0, 1, steps + 1, device=z.device)
    for k in range(steps):
        t = ts[k].expand(x.shape[0])
        with _amp():
            v = model(x, t).float()
        x = x + (ts[k + 1] - ts[k]) * v
        if keep_traj: traj.append(x.float().cpu())
    return (x, traj) if keep_traj else x


@torch.no_grad()
def ddim_sample(model, z, steps, abar, keep_traj=False):
    """Deterministic DDIM (eta = 0) on evenly spaced indices from 999 down to 0, then the x0 prediction."""
    idx = torch.linspace(T_DDPM - 1, 0, steps).round().long().tolist() if steps > 1 else [T_DDPM - 1]
    x = z.clone()
    traj = [x.float().cpu()] if keep_traj else None
    for k, i in enumerate(idx):
        a = abar[i]
        with _amp():
            eps = model(x, torch.full((x.shape[0],), i / T_DDPM, device=x.device)).float()
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        a_next = abar[idx[k + 1]] if k + 1 < len(idx) else torch.tensor(1.0, device=x.device)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()                    # re-derive eps after clipping x0
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        if keep_traj: traj.append(x.float().cpu())
    return (x, traj) if keep_traj else x


def path_stats(traj):
    """traj: list of (N, ...) tensors from start to end. Returns per-sample length/chord ratio
    (1.0 = straight line) and Liu et al.'s straightness S = mean_k ||(x_{k+1}-x_k)/dt - (x_end - x_start)||^2,
    normalised by ||x_end - x_start||^2 (0 = straight)."""
    X = torch.stack([t.flatten(1).double() for t in traj])            # (K+1, N, D)
    seg = X[1:] - X[:-1]
    chord = X[-1] - X[0]
    length = seg.norm(dim=-1).sum(0)
    K = seg.shape[0]
    S = ((seg * K - chord[None]) ** 2).sum(-1).mean(0) / (chord ** 2).sum(-1)
    return {"ratio": length / chord.norm(dim=-1), "S": S}
