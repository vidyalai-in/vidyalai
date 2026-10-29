"""Logit lens and tuned lens (Interpretability Part 2, writer p02).

Conventions
- Models load with HookedTransformer.from_pretrained_no_processing: no LayerNorm folding, no
  centering, so the residual stream is exactly the hidden state the Hugging Face model computes.
- "Point" l of the residual stream: 0 = embeddings (input to block 0), l = output of block l
  (= input to block l+1), n_layers = output of the last block (what the final norm reads).
- A lens maps a point's vector to next-token logits through the model's own final norm and
  unembedding: logit lens = unembed(ln_final(h)); tuned lens = unembed(ln_final(h + h W_l + b_l)).
"""
import json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def load(name, dtype=torch.float32, device="cuda"):
    model = HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def point_names(model):
    L = model.cfg.n_layers
    return [f"blocks.{l}.hook_resid_pre" for l in range(L)] + [f"blocks.{L-1}.hook_resid_post"]


@torch.no_grad()
def resid_points(model, tokens):
    """Return (final logits, stack of residual vectors with shape (n_layers+1, batch, pos, d_model))."""
    names = point_names(model)
    logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return logits, torch.stack([cache[n] for n in names])


def logit_lens(model, h):
    """Decode residual vectors h (..., d_model) with the model's final norm and unembedding."""
    return model.unembed(model.ln_final(h))


class TunedLens(nn.Module):
    """One affine translator per point 0..n_layers-1, initialised to the identity (W = 0, b = 0).

    kind = "full": h -> h + h W_l + b_l (Belrose et al. 2023, eq. 8, residual parameterisation)
    kind = "bias": h -> h + b_l          (the "debiased" logit lens, their eq. 7)
    The last point (the model's own output) needs no translator.
    """

    def __init__(self, n_layers, d_model, kind="full"):
        super().__init__()
        self.kind = kind
        self.n = n_layers
        self.bias = nn.Parameter(torch.zeros(n_layers, d_model))
        # one 2-D matrix per layer (Muon orthogonalises 2-D parameters)
        self.weight = (nn.ParameterList([nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_layers)])
                       if kind == "full" else None)

    def translate(self, l, h):
        if l >= self.n:
            return h
        out = h + self.bias[l]
        if self.weight is not None:
            out = out + h @ self.weight[l]
        return out


def kl_from_final(final_logp, lens_logits):
    """KL(final || lens) per token in nats. final_logp: log-probs (fp32); lens_logits: logits."""
    lq = F.log_softmax(lens_logits.float(), dim=-1)
    return (final_logp.exp() * (final_logp - lq)).sum(-1)


@torch.no_grad()
def evaluate(model, chunks, lenses=None, batch=8, skip_first=1):
    """Per-chunk, per-point mean KL(final || lens) in nats, top-1 agreement with the final argmax,
    cross-entropy against the real next token, and mean residual norm.

    chunks: LongTensor (n_chunks, T). lenses: dict name -> TunedLens (the logit lens is always run).
    Returns dict lens_name -> {"kl","top1","ce"} each a list [n_chunks][n_points], plus "final_ce",
    "norm" ([n_chunks][n_points], positions >= skip_first) and "norm_pos0" (position 0 only).
    """
    lenses = lenses or {}
    names = ["logit"] + list(lenses)
    P = model.cfg.n_layers + 1
    out = {n: {"kl": [], "top1": [], "ce": []} for n in names}
    out.update(final_ce=[], norm=[], norm_pos0=[])
    for i in range(0, len(chunks), batch):
        tok = chunks[i:i + batch].to(model.cfg.device)
        logits, H = resid_points(model, tok)
        out["norm"] += H[:, :, skip_first:].float().norm(dim=-1).mean(-1).T.tolist()
        out["norm_pos0"] += H[:, :, 0].float().norm(dim=-1).T.tolist()
        tgt = tok[:, skip_first + 1:]
        flp = F.log_softmax(logits[:, skip_first:-1].float(), -1)
        p = flp.exp()
        negent = (p * flp).sum(-1)                     # -H(final)
        ftop = flp.argmax(-1)
        out["final_ce"] += (-flp.gather(-1, tgt[..., None])[..., 0]).mean(1).tolist()
        del logits
        for n in names:
            kl = torch.zeros(tok.shape[0], P, device=tok.device); t1 = kl.clone(); ce = kl.clone()
            for l in range(P):
                h = H[l][:, skip_first:-1]
                if n != "logit":
                    h = lenses[n].translate(l, h)
                lq = F.log_softmax(logit_lens(model, h).float(), -1)
                kl[:, l] = (negent - (p * lq).sum(-1)).mean(1)
                t1[:, l] = (lq.argmax(-1) == ftop).float().mean(1)
                ce[:, l] = (-lq.gather(-1, tgt[..., None])[..., 0]).mean(1)
                del lq
            out[n]["kl"] += kl.tolist(); out[n]["top1"] += t1.tolist(); out[n]["ce"] += ce.tolist()
        del H, p, flp
    return out


def train_tuned_lens(model, batches, kind="full", opt="muon", lr=None, steps=None, layers=None,
                     eval_at=(), eval_fn=None, amp=torch.bfloat16, bias_lr=1e-3):
    """Train a tuned lens with the distillation loss sum_l KL(final || lens_l).

    batches: iterator of LongTensor (B, T) token chunks. Each layer's loss is backpropagated on its
    own, so only one layer's logits are in memory at a time. The loss is the soft-target
    cross-entropy, which equals KL(final || lens) plus the final distribution's entropy (a constant).
    opt: "muon" (torch.optim.Muon on the matrices, Adam on the biases), "sgd" (SGD with Nesterov
    momentum, grad clip 1, weight decay 1e-3: Belrose et al.'s original recipe) or "adam".
    Every optimizer uses a linear decay of the learning rate to 0 over `steps`.
    Returns (lens, log list).
    """
    L, d = model.cfg.n_layers, model.cfg.d_model
    dev = model.cfg.device
    lens = TunedLens(L, d, kind).to(dev)
    layers = list(range(L)) if layers is None else layers
    if opt == "sgd":
        o = [torch.optim.SGD(lens.parameters(), lr=lr or 1.0, momentum=0.9, nesterov=True, weight_decay=1e-3)]
    elif opt == "muon":
        o = [torch.optim.Adam([lens.bias], lr=bias_lr)]
        if lens.weight is not None:
            o.append(torch.optim.Muon(list(lens.weight), lr=lr or 0.02, weight_decay=0.0,
                                      adjust_lr_fn="match_rms_adamw"))
    else:
        o = [torch.optim.Adam(lens.parameters(), lr=lr or bias_lr)]
    sched = [torch.optim.lr_scheduler.LambdaLR(x, lambda s: max(0.0, 1 - s / steps)) for x in o]
    log, t0, ntok = [], time.time(), 0
    for step in range(steps):
        tok = next(batches).to(dev)
        logits, H = resid_points(model, tok)
        p = F.softmax(logits[:, 1:].float(), -1).flatten(0, 1)
        del logits
        total = 0.0
        for l in layers:
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                z = logit_lens(model, lens.translate(l, H[l][:, 1:])).flatten(0, 1)
                loss = F.cross_entropy(z, p)
            loss.backward()
            total += loss.item()
            del z, loss
        if opt == "sgd":
            torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        for x in o:
            x.step(); x.zero_grad(set_to_none=True)
        for s in sched:
            s.step()
        ntok += tok.numel()
        if (step + 1) in eval_at or step == steps - 1:
            rec = {"step": step + 1, "tokens": ntok, "train_soft_ce_mean": total / len(layers),
                   "sec": round(time.time() - t0, 1)}
            if eval_fn is not None:
                rec.update(eval_fn(lens))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    return lens, log


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Mean and 95% percentile bootstrap CI over independent units (here: documents)."""
    v = torch.as_tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(v), (n_boot, len(v)), generator=g)
    means = v[idx].mean(1)
    return float(v.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))
