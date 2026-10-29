"""Shared helpers for the Interpretability series (owner: Part 1).

load()           load a model with its precision and weight processing stated
run_cached()     run prompts (grouped by token length, so no padding) and keep a small cache
components()     split the last-position residual stream into the pieces that wrote it
dla()            direct logit attribution: each piece's dot product with a logit direction
ablate_hook()    a hook that zeroes or mean-replaces one activation (optionally one head)

Tested with transformer-lens 3.9.0 (HookedTransformer API), torch 2.11.
"""
from collections import defaultdict

import torch
from transformer_lens import HookedTransformer


def load(name, device="cuda", dtype=torch.float32, processing=True):
    """processing=True folds LayerNorm weights into the next matrix and centers the weights
    that write to the residual stream (TransformerLens defaults). False keeps the Hugging Face
    weights exactly, which is the only option for models too big to process in host RAM."""
    if processing:
        m = HookedTransformer.from_pretrained(name, device=device, dtype=dtype)
    else:
        m = HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=dtype)
    m.eval()
    m.setup_info = dict(name=name, dtype=str(dtype), processing=processing,
                        n_layers=m.cfg.n_layers, n_heads=m.cfg.n_heads, d_model=m.cfg.d_model,
                        normalization=m.cfg.normalization_type)
    return m


def length_groups(model, prompts):
    """Group prompt indices by token count so each batch needs no padding."""
    groups = defaultdict(list)
    toks = [model.to_tokens(p)[0] for p in prompts]
    for i, t in enumerate(toks):
        groups[len(t)].append(i)
    for n, idx in sorted(groups.items()):
        yield idx, torch.stack([toks[i] for i in idx])


def default_names(model):
    names = {"hook_embed", "ln_final.hook_scale"}
    if model.cfg.positional_embedding_type == "standard":
        names.add("hook_pos_embed")
    for l in range(model.cfg.n_layers):
        names |= {f"blocks.{l}.hook_attn_out", f"blocks.{l}.hook_mlp_out",
                  f"blocks.{l}.attn.hook_z", f"blocks.{l}.hook_resid_post"}
    return names


@torch.no_grad()
def run_cached(model, prompts, names=None, pos=-1):
    """Returns final-position logits [n, vocab] and a dict of final-position activations
    (each [n, ...], float32 on CPU-free GPU tensors), in the original prompt order."""
    names = names or default_names(model)
    out_logits, out = [None] * len(prompts), defaultdict(lambda: [None] * len(prompts))
    for idx, toks in length_groups(model, prompts):
        logits, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for j, i in enumerate(idx):
            out_logits[i] = logits[j, pos].float()
            for k in names:
                out[k][i] = cache[k][j, pos].float()
    return torch.stack(out_logits), {k: torch.stack(v) for k, v in out.items()}


def components(model, acts):
    """Pieces that add up to the final residual stream at the read position.
    Returns (labels, tensor [n_components, n_prompts, d_model])."""
    labels, parts = ["embed"], [acts["hook_embed"]]
    if "hook_pos_embed" in acts:
        labels.append("pos_embed"); parts.append(acts["hook_pos_embed"])
    for l in range(model.cfg.n_layers):
        labels += [f"attn{l}", f"mlp{l}"]
        parts += [acts[f"blocks.{l}.hook_attn_out"], acts[f"blocks.{l}.hook_mlp_out"]]
    return labels, torch.stack(parts)


def head_outputs(model, acts):
    """Per-head writes to the residual stream: z_h @ W_O[h]. [n_layers, n_heads, n, d_model].
    The output bias b_O is shared by the heads, so it is not part of any head."""
    out = []
    for l in range(model.cfg.n_layers):
        z = acts[f"blocks.{l}.attn.hook_z"]                        # [n, heads, d_head]
        out.append(torch.einsum("nhe,hed->hnd", z, model.W_O[l].float()))
    return torch.stack(out)


def to_logit_space(model, x, scale):
    """Apply the final norm as a fixed linear map (the scale is frozen from the clean run),
    so each piece's contribution is additive. x [..., n, d], scale [n, 1]."""
    if model.cfg.normalization_type in ("LN", "LNPre"):
        x = x - x.mean(-1, keepdim=True)
    x = x / scale
    w = getattr(model.ln_final, "w", None)                         # absent when folded
    if w is not None:
        x = x * w.float()
    return x


def dla(model, x, scale, direction):
    """Direct logit attribution. direction [n, d_model] is a per-prompt unembedding direction
    (e.g. W_U[:, answer] - mean of W_U[:, alternatives]). Returns [..., n]."""
    return (to_logit_space(model, x, scale) * direction).sum(-1)


def ablate_hook(mode, value=None, head=None, start=1):
    """mode 'zero' sets the activation to 0; 'mean' replaces it with `value`: one position's
    shape, or a callable seq_len -> [seq_len - start, ...] for position-dependent means. head selects one head of an [batch, pos, head, d_head] hook_z.
    Positions before `start` are left alone: with start=1 that is the BOS token, which is
    identical in every prompt."""
    def hook(act, hook):
        tgt = act[:, start:] if head is None else act[:, start:, head]
        if mode == "zero":
            tgt.zero_()
        else:
            v = value(act.shape[1]) if callable(value) else value   # callable: per-position means
            v = v if head is None else v[..., head, :]
            tgt.copy_(v.to(act.dtype).expand_as(tgt))
        return act
    return hook


@torch.no_grad()
def run_ablated(model, prompts, hook_name, hook_fn, pos=-1):
    """Final-position logits [n, vocab] with one hook applied."""
    out = [None] * len(prompts)
    for idx, toks in length_groups(model, prompts):
        logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook_fn)])
        for j, i in enumerate(idx):
            out[i] = logits[j, pos].float()
    return torch.stack(out)


def bootstrap_ci(x, n_boot=10000, seed=0, stat=None):
    """95% percentile bootstrap CI of the mean (or of stat) over prompts. x: 1-D tensor/array."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stat = stat or (lambda a: a.mean(-1))
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = stat(x[idx])
    return float(stat(x[None])[0]), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
