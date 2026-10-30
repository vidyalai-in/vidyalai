"""Training dynamics on the From Scratch checkpoints, with plain PyTorch hooks (owned by writer p10).

The From Scratch model (fromscratch/model.py: RMSNorm, RoPE, grouped-query attention, SwiGLU) calls
F.scaled_dot_product_attention, which never materializes attention weights. So the patterns are
recomputed here from each block's attention input: q and k from wq and wk, RoPE applied, each key-value
head shared by n_head // n_kv_head consecutive query heads (SDPA's enable_gqa uses repeat_interleave),
then a causal softmax. check_patterns() verifies the recomputation against the model's own output.

    from lab import dynamics as D
    model = D.load(D.snapshots("main88")[-1][1], D.RUNS["main88"].cfg)
    tok = D.repeated_tokens(200, 50, seed=1, allowed=D.frequent_tokens())
    s = D.head_scores(model, tok, half=50)        # {"prev","dup","induction": [n, L, H], "loss": [n, T-1]}

Nothing here writes outside the caller's own output paths; the run folders are only read.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SCRATCH = "/home/amit/.cache/vidyalai-scratch"
FS = f"{SCRATCH}/fromscratch"
if FS not in sys.path:
    sys.path.append(FS)
from model import GPT, MAIN, SMALL, apply_rope  # noqa: E402
from sizes import SIZES  # noqa: E402

from lab.heads import scores_from_pattern  # noqa: E402  (Part 4's score definitions)

EOT = 0
VAL_SHARD = f"{FS}/data/shards/val_000.bin"
TRAIN_SHARD0 = f"{FS}/data/shards/train_000.bin"
NGRAM = f"{SCRATCH}/p10/runs/ngram.npz"


@dataclass
class Run:
    path: str
    cfg: object
    tokens_per_step: int


RUNS = {
    "main88": Run(f"{SCRATCH}/t07/runs/main88", MAIN, 524_288),
    "s11": Run(f"{SCRATCH}/t08/runs/s11", SIZES["s11"], 65_536),
    "s15": Run(f"{SCRATCH}/t08/runs/s15", SIZES["s15"], 65_536),
    "s22": Run(f"{SCRATCH}/t08/runs/s22", SMALL, 131_072),
    "s33": Run(f"{SCRATCH}/t08/runs/s33", SIZES["s33"], 262_144),
    "s47": Run(f"{SCRATCH}/t08/runs/s47", SIZES["s47"], 262_144),
    "iso_s11": Run(f"{SCRATCH}/t08/runs/iso_s11", SIZES["s11"], 65_536),
    "iso_s22": Run(f"{SCRATCH}/t08/runs/iso_s22", SMALL, 131_072),
}


def snapshots(run: str):
    """[(step, path)] of a run's weights-only snapshots, sorted by step."""
    out = []
    for p in glob.glob(os.path.join(RUNS[run].path, "model_*.pt")):
        out.append((int(re.search(r"model_(\d+)\.pt", p).group(1)), p))
    return sorted(out)


def load(path: str, cfg, device: str = "cuda") -> GPT:
    """A From Scratch GPT with a snapshot's weights, in FP32, in eval mode (read-only on the file)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = GPT(cfg).to(device)
    model.load_state_dict(state.get("model", state))
    return model.eval()


# ---------------------------------------------------------------- hooks

class Hooks:
    """Context manager over plain forward pre-hooks.

    record=True keeps each block's attention input (the attn_norm output) as self.attn_in[layer], and
    the per-head attention output that feeds wo as self.z[layer] ([B, T, n_head, head_dim]).
    ablate={layer: (heads, means)} replaces those heads' wo input with means[head] (mean ablation,
    Part 4's choice), or with zeros if means is None."""

    def __init__(self, model: GPT, record: bool = True, ablate: dict | None = None):
        self.model, self.record, self.ablate = model, record, ablate or {}
        self.attn_in, self.z, self._h = {}, {}, []

    def __enter__(self):
        for l, blk in enumerate(self.model.blocks):
            if self.record:
                self._h.append(blk.attn.register_forward_pre_hook(self._grab_in(l)))
            self._h.append(blk.attn.wo.register_forward_pre_hook(self._wo_in(l)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []

    def _grab_in(self, l):
        def hook(mod, args):
            self.attn_in[l] = args[0].detach()
        return hook

    def _wo_in(self, l):
        nh, hd = self.model.cfg.n_head, self.model.cfg.head_dim

        def hook(mod, args):
            y = args[0]
            if self.record:
                self.z[l] = y.detach().view(*y.shape[:2], nh, hd)
            if l in self.ablate:
                heads, means = self.ablate[l]
                y = y.clone().view(*y.shape[:2], nh, hd)
                for h in heads:
                    y[:, :, h] = 0.0 if means is None else means[h].to(y.dtype)
                return (y.view(*y.shape[:2], nh * hd),)
            return None
        return hook


def attention_pattern(model: GPT, layer: int, x: torch.Tensor) -> torch.Tensor:
    """[B, n_head, T, T] attention weights of one block, recomputed in FP32 from its input x."""
    attn, cfg = model.blocks[layer].attn, model.cfg
    B, T, _ = x.shape
    x = x.float()
    q = F.linear(x, attn.wq.weight.float()).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
    k = F.linear(x, attn.wk.weight.float()).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
    q, k = apply_rope(q, model.rope_cos, model.rope_sin), apply_rope(k, model.rope_cos, model.rope_sin)
    k = k.repeat_interleave(cfg.n_head // cfg.n_kv_head, dim=1)        # GQA: head h reads kv head h // group
    s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    return s.masked_fill(mask, float("-inf")).softmax(-1)


@torch.no_grad()
def check_patterns(model: GPT, tokens: torch.Tensor) -> float:
    """Max |recomputed pattern @ V - the model's own attention output| over all layers (FP32 run)."""
    cfg = model.cfg
    with Hooks(model) as h:
        model(tokens)
    worst = 0.0
    for l in range(cfg.n_layer):
        x = h.attn_in[l].float()
        B, T, _ = x.shape
        v = F.linear(x, model.blocks[l].attn.wv.weight.float()).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        v = v.repeat_interleave(cfg.n_head // cfg.n_kv_head, dim=1)
        y = (attention_pattern(model, l, x) @ v).transpose(1, 2)        # [B, T, n_head, hd]
        worst = max(worst, (y - h.z[l].float()).abs().max().item())
    return worst


# ---------------------------------------------------------------- measurements

@torch.no_grad()
def head_scores(model: GPT, tokens: torch.Tensor, half: int | None, batch: int = 50, ablate=None):
    """Part 4's per-sequence head scores (lab.heads.scores_from_pattern) for every layer and query head,
    plus per-token loss. Returns {"prev", ["dup", "induction"]: [n, L, H], "loss": [n, T-1]}."""
    acc, losses = {}, []
    for s in range(0, tokens.shape[0], batch):
        tok = tokens[s:s + batch]
        with Hooks(model, ablate=ablate) as h:
            logits = model(tok)
        losses.append(F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tok[:, 1:].flatten(),
                                      reduction="none").view(tok.shape[0], -1).cpu())
        per = [scores_from_pattern(attention_pattern(model, l, h.attn_in[l]), half) for l in range(model.cfg.n_layer)]
        for k in per[0]:
            acc.setdefault(k, []).append(torch.stack([p[k] for p in per], 1).cpu())
    out = {k: torch.cat(v).numpy() for k, v in acc.items()}
    out["loss"] = torch.cat(losses).numpy()
    return out


@torch.no_grad()
def token_losses(model: GPT, tokens: torch.Tensor, batch: int = 16, bf16: bool = True, ablate=None) -> np.ndarray:
    """Per-token next-token loss [n, T-1] in nats. bf16=True runs under BF16 autocast, as training did."""
    out = []
    for s in range(0, tokens.shape[0], batch):
        tok = tokens[s:s + batch]
        with Hooks(model, record=False, ablate=ablate), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(tok)
        out.append(F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tok[:, 1:].flatten(),
                                   reduction="none").view(tok.shape[0], -1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def head_means(model: GPT, tokens: torch.Tensor, batch: int = 16) -> dict:
    """Mean wo input of every head over batch and position: {layer: [n_head, head_dim]}."""
    sums, n = {}, 0
    for s in range(0, tokens.shape[0], batch):
        with Hooks(model) as h:
            model(tokens[s:s + batch])
        for l, z in h.z.items():
            sums[l] = sums.get(l, 0) + z.float().sum((0, 1))
        n += z.shape[0] * z.shape[1]
    return {l: v / n for l, v in sums.items()}


def ablation(heads, means):
    """{layer: (heads, means[layer])} for Hooks(ablate=...) from a list of (layer, head)."""
    out = {}
    for l, hh in heads:
        out.setdefault(l, ([], None if means is None else means[l]))[0].append(hh)
    return out


# ---------------------------------------------------------------- data

def _shard(path):
    from data import read_shard
    return read_shard(path)


def frequent_tokens(min_count: int = 1000) -> np.ndarray:
    """Token ids seen at least min_count times in the first 100M training tokens (ngram.npz),
    excluding the 8 special tokens (ids 0-7). Random repeated sequences draw from these."""
    uni = np.load(NGRAM)["uni"]
    ids = np.where(uni >= min_count)[0]
    return ids[ids >= 8]


def repeated_tokens(n_seq: int, half: int, seed: int, allowed: np.ndarray, device="cuda") -> torch.Tensor:
    """[EOT] + r + r for n_seq random sequences r of `half` tokens drawn uniformly from `allowed`.
    <|endoftext|> precedes every document in training, so it plays the role of a BOS token."""
    g = np.random.default_rng(seed)
    r = torch.as_tensor(allowed[g.integers(0, len(allowed), (n_seq, half))], dtype=torch.long)
    b = torch.full((n_seq, 1), EOT, dtype=torch.long)
    return torch.cat([b, r, r], 1).to(device)


def val_documents(min_len: int):
    """(start, length) of every validation document with at least min_len tokens; start is the
    position of the <|endoftext|> token that precedes it."""
    v = _shard(VAL_SHARD)
    eot = np.flatnonzero(np.asarray(v) == EOT)
    lens = np.diff(np.append(eot, len(v))) - 1
    keep = lens >= min_len
    return eot[keep], lens[keep]


def val_windows(n: int, T: int, seed: int = 0, device="cuda") -> torch.Tensor:
    """n windows of T tokens, each = <|endoftext|> + the first T-1 tokens of a distinct validation
    document long enough to fill it, chosen at random (seeded)."""
    v = _shard(VAL_SHARD)
    starts, _ = val_documents(T - 1)
    pick = np.sort(np.random.default_rng(seed).choice(len(starts), n, replace=False))
    w = np.stack([np.asarray(v[starts[i]:starts[i] + T], dtype=np.int64) for i in pick])
    return torch.as_tensor(w, device=device)


# ---------------------------------------------------------------- n-gram statistics

class NGram:
    """Unigram and bigram distributions from ngram.npz, on the GPU, for KL(n-gram || model)."""

    def __init__(self, path: str = NGRAM, device="cuda"):
        d = np.load(path)
        uni = d["uni"].astype(np.float64)
        self.V = len(uni)
        self.p_uni = torch.tensor(uni / uni.sum(), device=device, dtype=torch.float32)
        pu = self.p_uni[self.p_uni > 0]
        self.negH_uni = float((pu * pu.log()).sum())
        crow, col, cnt = d["crow"], d["col"], d["cnt"].astype(np.float64)
        rows = np.repeat(np.arange(self.V), np.diff(crow))
        tot = np.bincount(rows, weights=cnt, minlength=self.V)
        p = cnt / tot[rows]
        negH = np.bincount(rows, weights=p * np.log(p), minlength=self.V)
        self.crow = torch.tensor(crow, device=device)
        self.col = torch.tensor(col, device=device, dtype=torch.long)
        self.p = torch.tensor(p, device=device, dtype=torch.float32)
        self.negH_bi = torch.tensor(negH, device=device, dtype=torch.float32)
        self.row_total = tot

    def dense_rows(self, a: torch.Tensor) -> torch.Tensor:
        """[N, V] bigram distributions P(. | a) for previous tokens a [N]."""
        st, ln = self.crow[a], self.crow[a + 1] - self.crow[a]
        rows = torch.repeat_interleave(torch.arange(len(a), device=a.device), ln)
        base = torch.repeat_interleave(st - (torch.cumsum(ln, 0) - ln), ln)
        idx = torch.arange(int(ln.sum()), device=a.device) + base
        out = torch.zeros(len(a), self.V, device=a.device)
        out[rows, self.col[idx]] = self.p[idx]
        return out

    @torch.no_grad()
    def kl(self, model: GPT, tokens: torch.Tensor, batch: int = 8, chunk: int = 1024):
        """Per-position KL(unigram || model) and KL(bigram || model) [n, T-1] in nats, on real text.
        The bigram row is P(next | current token); positions whose token never occurred in the
        counts are NaN."""
        ku, kb = [], []
        for s in range(0, tokens.shape[0], batch):
            tok = tokens[s:s + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tok)
            lq = logits[:, :-1].float().log_softmax(-1).flatten(0, 1)      # [N, V]
            a = tok[:, :-1].flatten()
            ku.append((self.negH_uni - lq @ self.p_uni).view(tok.shape[0], -1).cpu())
            out = torch.empty(len(a), device=a.device)
            for c in range(0, len(a), chunk):
                P = self.dense_rows(a[c:c + chunk])
                out[c:c + chunk] = self.negH_bi[a[c:c + chunk]] - (P * lq[c:c + chunk]).sum(-1)
            out[torch.as_tensor(self.row_total, device=a.device)[a] == 0] = float("nan")
            kb.append(out.view(tok.shape[0], -1).cpu())
        return torch.cat(ku).numpy(), torch.cat(kb).numpy()

    def sample(self, order: int, n: int, T: int, seed: int) -> np.ndarray:
        """n sequences of T tokens from the unigram (order 1) or bigram (order 2) model, each starting
        with <|endoftext|>: Belrose et al.'s maximum-entropy n-gram sequences."""
        g = np.random.default_rng(seed)
        p_uni = self.p_uni.cpu().double().numpy(); p_uni /= p_uni.sum()
        out = np.zeros((n, T), dtype=np.int64)
        if order == 1:
            out[:, 1:] = g.choice(self.V, size=(n, T - 1), p=p_uni)
            return out
        crow, col, p = self.crow.cpu().numpy(), self.col.cpu().numpy(), self.p.cpu().double().numpy()
        for i in range(n):
            a = EOT
            for t in range(1, T):
                lo, hi = crow[a], crow[a + 1]
                c = np.cumsum(p[lo:hi])
                a = int(col[lo + min(np.searchsorted(c, g.random() * c[-1]), hi - lo - 1)])
                out[i, t] = a
        return out


def ci95(x, axis=0):
    """Mean and normal-approximation 95% CI half-width."""
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis), 1.96 * x.std(axis, ddof=1) / np.sqrt(x.shape[axis])
