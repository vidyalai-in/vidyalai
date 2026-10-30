"""Robustness audit helpers for Interpretability Part 12 (owner: p12).

Re-runs headline measurements from Parts 1, 4 and 7 under a changed condition
(new seed, new prompt template, new text distribution) and reports how far they moved.

landmark_prompts()   Part 1's 168 landmark facts under any template with one "{}" slot
position_classes()   per-position class ids for template-aware mean ablation
layer_audit()        DLA and mean-ablation cost of every attention/MLP output (Part 1's method)
rank_agreement()     Spearman correlation + top-k overlap between two per-component scores, with a bootstrap CI
repeat_natural()     [BOS] + s + s windows cut from real text, for the induction test on natural tokens
frequent_ids()       token ids seen at least `min_count` times in a token array
windows_from()       cut long token rows into BOS-prefixed windows (for SAE evaluation)
"""
from __future__ import annotations

import numpy as np
import torch

from lab import hooks as H


# ---------------------------------------------------------------- Part 1: landmark recall
def landmark_prompts(model, facts, template, lower_the=False):
    """facts: [(landmark, city)]. Keeps facts whose city is one token for this model.
    lower_the: write a leading 'The ' as 'the ' (for templates where the landmark is mid-sentence)."""
    tok1 = lambda c: model.to_tokens(" " + c, prepend_bos=False)[0]
    keep = [(l, c) for l, c in facts if len(tok1(c)) == 1]
    fmt = lambda l: ("the " + l[4:]) if (lower_the and l.startswith("The ")) else l
    prompts = [template.format(fmt(l)) for l, _ in keep]
    cities = sorted({c for _, c in keep})
    dev = model.cfg.device
    cid = torch.tensor([tok1(c).item() for c in cities], device=dev)
    ans = torch.tensor([cities.index(c) for _, c in keep], device=dev)
    return keep, prompts, cities, cid, ans


def city_metrics(logits, cid, ans):
    """Logit difference (answer minus mean of the other candidate cities), top-1 over the
    vocabulary, top-1 among the candidate cities."""
    cl = logits[:, cid]
    a = cl.gather(1, ans[:, None])[:, 0]
    others = (cl.sum(1) - a) / (cl.shape[1] - 1)
    return a - others, logits.argmax(1) == cid[ans], cl.argmax(1) == ans


def position_classes(model, template, start):
    """Returns cls(T) -> LongTensor[T - start]: one class per template token before the slot
    (counted from the start), one class for all landmark tokens (class 0), and one class per
    template token after the slot (counted from the end). Prompts of equal length share it."""
    pre, post = template.split("{}")
    P = len(model.to_tokens(pre, prepend_bos=False)[0]) if pre else 0
    S = len(model.to_tokens(post, prepend_bos=False)[0])
    n_cls = 1 + P + S

    def cls(T):
        n = T - start
        c = torch.zeros(n, dtype=torch.long)
        c[:P] = torch.arange(1, P + 1)
        c[n - S:] = torch.arange(P + S, P, -1)          # distance from the end, offset past the prefix
        return c.to(model.cfg.device)
    return cls, n_cls, P, S


@torch.no_grad()
def layer_audit(model, prompts, cid, ans, template, start):
    """Part 1's two measurements for every attention and MLP output at the last position:
    direct logit attribution and the change in logit difference when the output is replaced by
    its mean (one mean per template position, one for the landmark tokens).
    Returns dict with clean metrics and per-component arrays [n_comp, n_prompts]."""
    logits, acts = H.run_cached(model, prompts)
    ld, top1, topc = city_metrics(logits, cid, ans)
    WU = model.W_U.float()[:, cid].T
    dirs = WU[ans] - (WU.sum(0, keepdim=True) - WU[ans]) / (len(cid) - 1)
    labels, comps = H.components(model, acts)
    dla = H.dla(model, comps, acts["ln_final.hook_scale"], dirs)
    keep = [i for i, l in enumerate(labels) if l.startswith(("attn", "mlp"))]
    labels = [labels[i] for i in keep]
    dla = dla[keep]

    cls, n_cls, _, _ = position_classes(model, template, start)
    names = [f"blocks.{l}.{k}" for l in range(model.cfg.n_layers) for k in ("hook_attn_out", "hook_mlp_out")]
    sums, counts = {}, torch.zeros(n_cls, device=model.cfg.device)

    def acc(act, hook):
        a = act[:, start:].float().sum(0)
        s = torch.zeros((n_cls,) + a.shape[1:], device=a.device).index_add_(0, cls(act.shape[1]), a)
        sums[hook.name] = sums.get(hook.name, 0) + s
    for idx, toks in H.length_groups(model, prompts):
        model.run_with_hooks(toks, fwd_hooks=[(n, acc) for n in names])
        counts.index_add_(0, cls(toks.shape[1]), torch.full((toks.shape[1] - start,), float(len(idx)), device=counts.device))
    means = {k: v / counts.clamp(min=1)[:, None] for k, v in sums.items()}

    abl = []
    for lab in labels:
        l = int(lab[4:] if lab.startswith("attn") else lab[3:])
        hname = f"blocks.{l}.hook_attn_out" if lab.startswith("attn") else f"blocks.{l}.hook_mlp_out"
        lg = H.run_ablated(model, prompts, hname, H.ablate_hook("mean", lambda T, h=hname: means[h][cls(T)], start=start))
        abl.append((city_metrics(lg, cid, ans)[0] - ld))
    return {"labels": labels, "ld": ld.cpu().numpy(), "top1": top1.cpu().numpy(), "topc": topc.cpu().numpy(),
            "dla": dla.cpu().numpy(), "d_mean": torch.stack(abl).cpu().numpy()}


def rank_agreement(a, b, k=5, n_boot=2000, seed=0):
    """a, b: [n_comp, n_prompts] per-prompt scores. Spearman correlation of the per-component
    means and the overlap of the top-k lists, each with a percentile bootstrap over prompts."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n = a.shape[1]

    def stat(idx):
        x, y = a[:, idx].mean(1), b[:, idx].mean(1)
        top = lambda v: set(np.argsort(-v)[:k])
        return spearmanr(x, y)[0], len(top(x) & top(y))
    pt = stat(np.arange(n))
    bs = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    ci = lambda j: [float(pt[j]), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
    return {"spearman": ci(0), f"top{k}_overlap": ci(1)}


def boot_mean(x, n_boot=10000, seed=0):
    return H.bootstrap_ci(np.asarray(x), n_boot=n_boot, seed=seed)


def boot_diff(x, y, n_boot=10000, seed=0):
    """CI of mean(x) - mean(y) for two independent samples (resampled separately)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    d = (x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1) - y[rng.integers(0, len(y), (n_boot, len(y)))].mean(1))
    return float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- Part 4: induction heads
def frequent_ids(token_array, min_count=5, top=None, skip_first=0):
    """Token ids seen at least min_count times; or the `top` most frequent after skipping the
    `skip_first` most frequent (e.g. skip punctuation and 'the')."""
    ids, c = np.unique(np.asarray(token_array).ravel(), return_counts=True)
    if top is None:
        return ids[c >= min_count]
    order = np.argsort(-c)
    return ids[order[skip_first:skip_first + top]]


def repeat_natural(token_array, n_seq, half, bos, seed, device="cuda"):
    """[BOS] + s + s where s is a random `half`-token slice of real text (one per row)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(token_array)
    rows = rng.choice(len(arr), n_seq, replace=len(arr) < n_seq)
    offs = rng.integers(0, arr.shape[1] - half + 1, n_seq)
    s = torch.tensor(np.stack([arr[r, o:o + half] for r, o in zip(rows, offs)]), dtype=torch.long)
    b = torch.full((n_seq, 1), bos)
    return torch.cat([b, s, s], 1).to(device)


# ---------------------------------------------------------------- Part 7: SAE off-distribution
def windows_from(token_array, width, bos, group=None):
    """Cut rows of a [n, L] token array into non-overlapping windows of `width` tokens and put
    BOS in front: [m, width + 1]. group: optional per-row id (e.g. article), returned per window."""
    arr = np.asarray(token_array)
    k = arr.shape[1] // width
    w = arr[:, :k * width].reshape(arr.shape[0] * k, width)
    out = torch.cat([torch.full((w.shape[0], 1), bos), torch.tensor(w, dtype=torch.long)], 1)
    g = None if group is None else np.repeat(np.asarray(group), k)
    return out, g
