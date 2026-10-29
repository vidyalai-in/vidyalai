"""Attention-head scores and head ablation (owned by writer p04).

Scores follow Olsson et al. 2022 ("In-context Learning and Induction Heads"):
  - previous-token score: mean attention from token i to token i-1
  - duplicate-token score: on a repeated random sequence, mean attention from a
    second-half token to the earlier copy of the same token
  - induction (prefix-matching) score: mean attention from a second-half token to
    the token right AFTER its earlier copy
All scores are computed per sequence, so callers can put confidence intervals on them.
"""
from __future__ import annotations

import numpy as np
import torch
import transformer_lens as tl


def load(name: str, device: str = "cuda", dtype=torch.float32, processing: bool = True, **kw):
    """Load a HookedTransformer. processing=True folds LayerNorm and centers writing
    weights (only use it for small models: it needs a full fp32 copy on the host)."""
    if processing:
        return tl.HookedTransformer.from_pretrained(name, device=device, dtype=dtype, **kw)
    return tl.HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=dtype, **kw)


def repeated_tokens(model, n_seq: int, half: int, seed: int, skip: int = 1000, allowed=None) -> torch.Tensor:
    """[BOS] + r + r for n_seq random sequences r of `half` tokens.
    Token ids are drawn uniformly from [skip, vocab - skip), which drops the most
    common byte-level tokens at the start of BPE vocabularies and the rarest merges
    at the end (Olsson et al. also exclude the most and least common tokens)."""
    g = torch.Generator().manual_seed(seed)
    vocab = model.tokenizer.vocab_size
    if allowed is None:
        r = torch.randint(skip, vocab - skip, (n_seq, half), generator=g)
    else:  # draw from a given set of token ids (e.g. tokens seen often enough in real text)
        allowed = torch.as_tensor(allowed)
        r = allowed[torch.randint(0, len(allowed), (n_seq, half), generator=g)]
    bos = model.tokenizer.bos_token_id
    if bos is None:  # Qwen2.5 has no BOS; use its end-of-text token as a separator
        bos = model.tokenizer.eos_token_id
    b = torch.full((n_seq, 1), bos)
    return torch.cat([b, r, r], dim=1).to(model.cfg.device)


def _pattern_hooks(model, store):
    def hook(pattern, hook):  # pattern: [batch, head, dest, src]
        store[hook.layer()] = pattern.detach()
    return [(f"blocks.{l}.attn.hook_pattern", hook) for l in range(model.cfg.n_layers)]


def scores_from_pattern(pattern: torch.Tensor, half: int | None) -> dict:
    """Per-sequence, per-head scores from one layer's pattern [batch, head, dest, src].
    half=None means natural text: only the previous-token score is defined."""
    T = pattern.shape[-1]
    i = torch.arange(1, T, device=pattern.device)
    out = {"prev": pattern[:, :, i, i - 1].mean(-1)}  # skip dest 0 (BOS)
    if half is not None:
        d = torch.arange(half + 1, 2 * half + 1, device=pattern.device)  # second copy
        out["dup"] = pattern[:, :, d, d - half].mean(-1)
        out["induction"] = pattern[:, :, d, d - half + 1].mean(-1)
    return out


@torch.no_grad()
def head_scores(model, tokens: torch.Tensor, half: int | None, batch: int = 32, fwd_hooks=()):
    """Returns {score: array [n_seq, n_layers, n_heads]} plus per-token loss [n_seq, T-1]."""
    acc = {}
    losses = []
    for s in range(0, tokens.shape[0], batch):
        tok = tokens[s:s + batch]
        store = {}
        logits = model.run_with_hooks(tok, fwd_hooks=list(fwd_hooks) + _pattern_hooks(model, store))
        lp = torch.log_softmax(logits.float(), -1)
        losses.append((-lp[:, :-1].gather(-1, tok[:, 1:, None])[..., 0]).cpu())
        per = [scores_from_pattern(store[l].float(), half) for l in range(model.cfg.n_layers)]
        for k in per[0]:
            acc.setdefault(k, []).append(torch.stack([p[k] for p in per], 1).cpu())
    res = {k: torch.cat(v).numpy() for k, v in acc.items()}
    res["loss"] = torch.cat(losses).numpy()
    return res


@torch.no_grad()
def mean_z(model, tokens: torch.Tensor, batch: int = 32) -> dict:
    """Mean of each head's z (its value-weighted output before W_O) over batch and position."""
    sums, n = {}, 0
    for s in range(0, tokens.shape[0], batch):
        _, cache = model.run_with_cache(tokens[s:s + batch], names_filter=lambda x: x.endswith("hook_z"))
        for l in range(model.cfg.n_layers):
            z = cache[f"blocks.{l}.attn.hook_z"].float()  # [b, pos, head, d_head]
            sums[l] = sums.get(l, 0) + z.sum((0, 1))
        n += z.shape[0] * z.shape[1]
    return {l: (v / n) for l, v in sums.items()}


def ablate_heads_hooks(heads, means: dict | None):
    """Hooks that replace head outputs (hook_z) with their mean (means given) or zero."""
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        def hook(z, hook, hs=hs, l=l):
            for h in hs:
                z[:, :, h] = 0.0 if means is None else means[l][h].to(z.dtype)
            return z
        hooks.append((f"blocks.{l}.attn.hook_z", hook))
    return hooks


@torch.no_grad()
def token_loss(model, tokens: torch.Tensor, fwd_hooks=(), batch: int = 32) -> np.ndarray:
    """Per-token cross-entropy loss [n_seq, T-1] (nats)."""
    out = []
    for s in range(0, tokens.shape[0], batch):
        tok = tokens[s:s + batch]
        logits = model.run_with_hooks(tok, fwd_hooks=list(fwd_hooks))[:, :-1]
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), tok[:, 1:].flatten(), reduction="none")
        out.append(ce.view(tok.shape[0], -1).cpu())
    return torch.cat(out).numpy()


def top_heads(score: np.ndarray, k: int):
    """k (layer, head) pairs with the largest mean score; score is [n_seq, L, H] or [L, H]."""
    m = score.mean(0) if score.ndim == 3 else score
    idx = np.argsort(m.ravel())[::-1][:k]
    return [tuple(int(x) for x in np.unravel_index(i, m.shape)) for i in idx]


def ci95(x: np.ndarray, axis: int = 0):
    """Mean and normal-approximation 95% CI half-width along axis."""
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis), 1.96 * x.std(axis, ddof=1) / np.sqrt(x.shape[axis])


def k_composition(model, random_baseline: int = 200, seed: int = 0):
    """K-composition score ||W_QK^{h2} W_OV^{h1}||_F / (||W_QK^{h2}||_F ||W_OV^{h1}||_F)
    (Elhage et al. 2021), for every earlier head h1 and later head h2, as
    [L1, H1, L2, H2], plus the mean score of random Gaussian matrices of the same shapes."""
    L, Hn = model.cfg.n_layers, model.cfg.n_heads
    scores = np.zeros((L, Hn, L, Hn), dtype=np.float32)
    with torch.no_grad():  # one (earlier head, later layer) slice at a time keeps memory small
        for l1 in range(L):
            for h1 in range(Hn):
                ov = (model.W_V[l1, h1] @ model.W_O[l1, h1]).float()         # [d, d]
                for l2 in range(l1 + 1, L):
                    qk = (model.W_Q[l2] @ model.W_K[l2].transpose(-1, -2)).float()  # [H, d, d]
                    num = (ov[None] @ qk.transpose(-1, -2)).norm(dim=(-2, -1))
                    scores[l1, h1, l2] = (num / (ov.norm() * qk.norm(dim=(-2, -1)))).cpu().numpy()
    g = torch.Generator().manual_seed(seed)
    d, dh = model.cfg.d_model, model.cfg.d_head
    rnd = []
    for _ in range(random_baseline):
        a = [torch.randn(d, dh, generator=g) for _ in range(4)]
        ov = a[0] @ a[1].T   # rank-dh d x d, like W_V W_O
        qk = a[2] @ a[3].T   # like W_Q W_K^T
        rnd.append(((ov @ qk.T).norm() / (ov.norm() * qk.norm())).item())
    return scores, float(np.mean(rnd)), float(np.std(rnd))


@torch.no_grad()
def copying_score(model, layer: int, head: int) -> float:
    """sum(Re eigenvalues) / sum(|eigenvalues|) of the full OV circuit W_E W_OV W_U
    (TransformerLens row-vector convention). +1: every eigenvalue positive (copying)."""
    fm = model.W_E @ model.OV[layer, head] @ model.W_U
    ev = fm.eigenvalues
    return float(ev.real.sum() / ev.abs().sum())


def k_path_ablation_hooks(model, senders, receivers, means: dict):
    """Remove senders' output (minus its mean) from the KEY input of receiver heads only.
    Everything else (the senders' effect on queries, values and the rest of the model)
    is left intact. Needs model.set_use_split_qkv_input(True)."""
    store = {}
    hooks = []
    for l, h in senders:
        def grab(z, hook, l=l, h=h):
            store[(l, h)] = (z[:, :, h] - means[l][h].to(z.dtype)) @ model.W_O[l, h]
            return z
        hooks.append((f"blocks.{l}.attn.hook_z", grab))
    by_layer = {}
    for l, h in receivers:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def fix(k_in, hook, l=l, hs=hs):  # k_in: [batch, pos, head, d_model]
            for (sl, sh), delta in store.items():
                if sl < l:
                    for h in hs:
                        k_in[:, :, h] = k_in[:, :, h] - delta
            return k_in
        hooks.append((f"blocks.{l}.hook_k_input", fix))
    return hooks
