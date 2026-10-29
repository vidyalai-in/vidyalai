"""Activation patching, path patching and causal tracing (owner: Part 5).

Vocabulary (Heimersheim & Nanda 2024, arXiv 2404.15255):
  denoising  run the corrupted prompt, paste in activations cached from the clean prompt
             ("clean -> corrupt"): shows which activations are SUFFICIENT to restore the answer
  noising    run the clean prompt, paste in activations from the corrupted prompt
             ("corrupt -> clean"): shows which activations are NECESSARY for it

Every sweep returns per-prompt metric values, so callers can bootstrap confidence intervals.
Prompts in one call must have the same token length (group them first, e.g. hooks.length_groups);
which positions to patch is given per prompt as a boolean mask [batch, pos].

Tested with transformer-lens 3.9.0 (HookedTransformer API), torch 2.11.
"""
from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------- metrics

def logit_diff(logits, ans, alt):
    """logit(ans) - logit(alt) at the last position. logits [b, T, V] or [b, V]; ans, alt [b].
    alt may also be [b, k]: then the mean over the k alternatives is subtracted."""
    lg = logits[:, -1] if logits.dim() == 3 else logits
    a = lg.gather(1, ans[:, None])[:, 0]
    alt = alt if alt.dim() == 2 else alt[:, None]
    return (a - lg.gather(1, alt).mean(1)).float()


def prob(logits, ans):
    """Softmax probability of ans at the last position."""
    lg = logits[:, -1] if logits.dim() == 3 else logits
    return lg.float().softmax(-1).gather(1, ans[:, None])[:, 0]


def normalized(patched, clean, corrupt):
    """Share of the clean-corrupt gap that a patch restores, as a ratio of means:
    0 = no change from the corrupted run, 1 = back to the clean run (Wang et al. 2022;
    Zhang & Nanda 2023). Works on per-prompt arrays of any shape [..., n]."""
    return (patched.mean(-1) - corrupt.mean(-1)) / (clean.mean(-1) - corrupt.mean(-1))


def bootstrap_normalized(patched, clean, corrupt, n_boot=2000, seed=0):
    """95% percentile bootstrap CI over prompts for normalized(). patched [..., n]."""
    patched, clean, corrupt = (np.asarray(x, dtype=np.float64) for x in (patched, clean, corrupt))
    n = clean.shape[-1]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    p, c, k = patched[..., idx], clean[idx], corrupt[idx]          # [..., B, n]
    s = (p.mean(-1) - k.mean(-1)) / (c.mean(-1) - k.mean(-1))
    est = normalized(patched, clean, corrupt)
    return est, np.percentile(s, 2.5, axis=-1), np.percentile(s, 97.5, axis=-1)


def bootstrap_mean(x, n_boot=2000, seed=0):
    """Mean and 95% percentile bootstrap CI over the last axis."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, x.shape[-1], size=(n_boot, x.shape[-1]))
    s = x[..., idx].mean(-1)
    return x.mean(-1), np.percentile(s, 2.5, axis=-1), np.percentile(s, 97.5, axis=-1)


# ---------------------------------------------------------------- hooks

def replace_hook(source, mask, head=None):
    """Overwrite act[mask] with source[mask]. act/source: [b, T, ...]; mask [b, T] bool.
    head selects one head of a per-head hook ([b, T, n_heads, d], e.g. attn.hook_z)."""
    def hook(act, hook):
        if head is None:
            act[mask] = source[mask].to(act.dtype)
        else:
            sel = act[:, :, head]                                  # a view into act
            sel[mask] = source[:, :, head][mask].to(act.dtype)
        return act
    return hook


def add_hook(delta, mask):
    """act[mask] += delta[mask] (delta [b, T, d]); used for Gaussian-noise corruption."""
    def hook(act, hook):
        act[mask] = act[mask] + delta[mask].to(act.dtype)
        return act
    return hook


def cache_acts(model, toks, names, fwd_hooks=()):
    """Run once and keep the named activations (full sequence). Returns (logits, {name: act})."""
    store = {}

    def grab(act, hook):
        store[hook.name] = act.detach().clone()
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=list(fwd_hooks) + [(n, grab) for n in names])
    return logits, store


def run(model, toks, fwd_hooks):
    """Logits at the LAST position only, [b, vocab]: stops after the final block and unembeds one
    position, which saves the unembedding of every other position (a large share of the cost
    for small models with big vocabularies)."""
    with torch.no_grad():
        resid = model.run_with_hooks(toks, fwd_hooks=list(fwd_hooks), stop_at_layer=model.cfg.n_layers)
        return model.unembed(model.ln_final(resid[:, -1:]))[:, 0]


# ---------------------------------------------------------------- sweeps

def sweep(model, dest_toks, source_cache, items, metric, extra_hooks=()):
    """Generic exploratory sweep. items: list of (label, [(hook_name, mask, head_or_None), ...]);
    each item patches the listed activations from source_cache into a run on dest_toks
    (all at once) and records metric(logits) per prompt. extra_hooks (e.g. the noise that
    makes a corrupted run) are applied first in every run. Returns {label: np.array [b]}."""
    out = {}
    for label, patches in items:
        hooks = list(extra_hooks) + [(n, replace_hook(source_cache[n], m, h)) for n, m, h in patches]
        out[label] = metric(run(model, dest_toks, hooks)).cpu().numpy()
    return out


# ---------------------------------------------------------------- path patching

def path_patch(model, orig_toks, orig_z, new_z, sender, receivers, metric, sender_mask=None):
    """Path patching as in Wang et al. (2022), Appendix B: the effect of sender head (l, h)
    on a set of receivers, through the residual stream and MLPs but NOT through other heads.

    Run 2: run orig_toks with every head's z frozen to orig_z, except the sender's, which takes
    new_z (all positions); MLPs and layer norms are recomputed. Run 3: run orig_toks again with
    only the receivers' inputs replaced by their run-2 values.
    orig_z / new_z: {layer: z [b, T, n_heads, d_head]}. sender_mask [b, T] limits the sender's
    patch to some positions (None: all positions).
    receivers: "logits" (the direct path to the output: run 2's logits), or a list of
    (layer, head, kind, mask) with kind in {"q", "k", "v"} (needs model.cfg.use_split_qkv_input)
    and mask [b, T] the receiver positions. Returns metric(run-3 logits) per prompt."""
    sl, sh = sender

    def freeze(z, hook):
        l = hook.layer()
        z[:] = orig_z[l].to(z.dtype)
        if l == sl:
            if sender_mask is None:
                z[:, :, sh] = new_z[l][:, :, sh].to(z.dtype)
            else:
                sel = z[:, :, sh]
                sel[sender_mask] = new_z[l][:, :, sh][sender_mask].to(z.dtype)
        return z
    hooks = [(f"blocks.{l}.attn.hook_z", freeze) for l in range(model.cfg.n_layers)]
    if receivers == "logits":
        return metric(run(model, orig_toks, hooks)).cpu().numpy()
    names = sorted({f"blocks.{l}.hook_{k}_input" for l, h, k, m in receivers})
    _, rec = cache_acts(model, orig_toks, names, fwd_hooks=hooks)
    hooks3 = [(f"blocks.{l}.hook_{k}_input", replace_hook(rec[f"blocks.{l}.hook_{k}_input"], m, h))
              for l, h, k, m in receivers]
    return metric(run(model, orig_toks, hooks3)).cpu().numpy()


# ---------------------------------------------------------------- causal tracing (Meng et al. 2022)

def embedding_noise(model, toks, subject_mask, sigma_mult=3.0, n_samples=10, seed=0, sigma=None):
    """Gaussian-noise corruption of the subject's token embeddings (ROME's causal tracing):
    noise ~ N(0, nu) with nu = sigma_mult x the standard deviation of the subject tokens'
    embeddings. Returns (repeated toks [b*n, T], repeated mask, noise [b*n, T, d], sigma).
    Each prompt is repeated n_samples times with a different noise draw."""
    if sigma is None:
        emb = model.W_E[toks[subject_mask]].float()
        sigma = emb.std().item()
    g = torch.Generator(device="cpu").manual_seed(seed)
    b, T = toks.shape
    r_toks = toks.repeat_interleave(n_samples, 0)
    r_mask = subject_mask.repeat_interleave(n_samples, 0)
    noise = torch.randn((b * n_samples, T, model.cfg.d_model), generator=g) * (sigma_mult * sigma)
    return r_toks, r_mask, noise.to(model.cfg.device), sigma


def window(layer, size, n_layers):
    """Layers restored together for a window centred on `layer` (ROME: [l-4, l+5] for size 10)."""
    lo = layer - (size - 1) // 2
    return [l for l in range(lo, lo + size) if 0 <= l < n_layers]
