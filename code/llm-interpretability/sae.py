"""Sparse autoencoders: training on model activations, evaluation, Gemma Scope loading
(Interpretability Part 7, writer p07).

SAE(d, n, kind)       kind "relu" (L1 penalty weighted by decoder norms, Anthropic's April 2024 recipe),
                      "topk" (Gao et al. 2024, with the AuxK loss for dead latents) or "jumprelu"
                      (inference only, for Gemma Scope's released weights).
gpt2_chunks()         stream GPT-2 activations at one hook in shuffled chunks (BOS positions dropped).
train()               train several SAEs side by side on the same activation stream, with checkpoints.
evaluate_gpt2()       L0, fraction of variance explained, and next-token loss with the SAE spliced in.
TopK tracker          running top-k activating (window, position) pairs per latent.
load_gemma_scope()    download (if needed) and load one Gemma Scope params.npz as a JumpReLU SAE.
GemmaRunner           Gemma 2 through plain transformers: read one layer's output, stop early, or splice.

Every SAE stores a scalar `scale`; it divides inputs by it so that E[|x/scale|^2] = d_model during training,
and multiplies reconstructions back, so callers always see raw activation units.
Tested with torch 2.11, transformer-lens 3.9.0, transformers 5.21.1, sae-lens 6.54.12.
"""
import math, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, d, n, kind="relu", k=None, l1=0.0, seed=0, dec_init_norm=0.1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(n, d, generator=g)
        W = W / W.norm(dim=1, keepdim=True) * (dec_init_norm if kind == "relu" else 1.0)
        self.W_dec = nn.Parameter(W)                      # [n, d]: row i is latent i's direction
        self.W_enc = nn.Parameter(W.t().clone())          # [d, n]
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))         # TopK: b_pre, subtracted before encoding
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("threshold", torch.zeros(n))  # JumpReLU only
        self.kind, self.k, self.l1, self.d, self.n = kind, k, l1, d, n

    def pre(self, x):
        x = x / self.scale
        if self.kind == "topk":
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def act(self, pre):
        if self.kind == "relu":
            return F.relu(pre)
        if self.kind == "jumprelu":
            return pre * (pre > self.threshold)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, F.relu(v))

    def encode(self, x):
        return self.act(self.pre(x))

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) * self.scale

    def forward(self, x):
        pre = self.pre(x)
        f = self.act(pre)
        return self.decode(f), f, pre


def loss_fn(sae, x, dead_mask=None, l1_now=None, k_aux=512, aux_coef=1 / 32):
    """x in raw units. Returns (loss, parts dict). Reconstruction error is measured in normalized units,
    as the mean over tokens of the squared error summed over dimensions."""
    xn = x / sae.scale
    pre = sae.pre(x)
    f = sae.act(pre)
    xh = f @ sae.W_dec + sae.b_dec
    err = xn - xh
    mse = err.pow(2).sum(-1).mean()
    active = f > 0
    parts = {"mse": mse.detach(), "l0": active.float().sum(-1).mean(), "fired": active.any(0)}
    loss = mse
    if sae.kind == "relu":
        sp = (f @ sae.W_dec.norm(dim=1)).mean()             # sum_i f_i |W_dec,i|, averaged over tokens
        loss = loss + (sae.l1 if l1_now is None else l1_now) * sp
        parts["sparsity"] = sp.detach()
    elif sae.kind == "topk" and dead_mask is not None and dead_mask.any():
        kk = min(k_aux, int(dead_mask.sum()))
        p = pre.masked_fill(~dead_mask, -float("inf"))
        v, i = p.topk(kk, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, i, F.relu(v))
        aux = (err.detach() - z @ sae.W_dec).pow(2).sum(-1).mean()
        loss = loss + aux_coef * aux
        parts["aux"] = aux.detach()
    return loss, parts


@torch.no_grad()
def gpt2_chunks(model, tokens, layer, seqs_per_chunk=4096, sub=256, start=0, seed=0):
    """Yield (chunk_index, activations [n_tok, d]) at blocks.{layer}.hook_resid_pre, BOS dropped, shuffled."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    for c, i in enumerate(range(start * seqs_per_chunk, len(tokens), seqs_per_chunk), start=start):
        acts = []
        for j in range(i, min(i + seqs_per_chunk, len(tokens)), sub):
            t = tokens[j:j + sub].to(torch.int64).cuda()
            h = model(t, stop_at_layer=layer)                     # = blocks.{layer}.hook_resid_pre
            acts.append(h[:, 1:].reshape(-1, h.shape[-1]))
        A = torch.cat(acts)
        yield c, A[torch.randperm(len(A), device="cuda", generator=g)]


def make_optim(sae, lr):
    return torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))


def lr_factor(step, total, warm=0.01, decay_from=0.8):
    if step < warm * total:
        return (step + 1) / (warm * total)
    if step > decay_from * total:
        return max(0.0, (total - step) / ((1 - decay_from) * total))
    return 1.0


def train(saes, lrs, chunks, total_steps, batch=4096, ckpt=None, log=None, ckpt_every=1000,
          dead_window_tokens=4_000_000, l1_warm=0.05, state=None, time_limit=None):
    """Train SAEs (dict name -> SAE) side by side; every batch goes to all of them.
    state (from a checkpoint) resumes optimizers, step counters and last-fired steps."""
    opts = {k: make_optim(s, lrs[k]) for k, s in saes.items()}
    last = {k: torch.zeros(s.n, dtype=torch.long, device="cuda") for k, s in saes.items()}  # last step fired
    step = 0
    if state:
        for k in saes:
            opts[k].load_state_dict(state["opt"][k]); last[k] = state["last"][k].cuda()
        step = state["step"]
    dead_steps = dead_window_tokens // batch
    t0 = time.time()
    hist = []
    for c, A in chunks:
        for b in range(0, len(A) - batch + 1, batch):
            if step >= total_steps:
                break
            x = A[b:b + batch]
            rec = {"step": step}
            for k, s in saes.items():
                for gp in opts[k].param_groups:
                    gp["lr"] = lrs[k] * lr_factor(step, total_steps)
                l1_now = s.l1 * min(1.0, step / (l1_warm * total_steps)) if s.kind == "relu" else None
                dead = (step - last[k]) > dead_steps if s.kind == "topk" else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(s.kind == "topk")):  # ReLU: TF32 was faster
                    loss, parts = loss_fn(s, x, dead_mask=dead, l1_now=l1_now)
                opts[k].zero_grad(set_to_none=True)
                loss.backward()
                if s.kind == "topk":                      # keep decoder rows unit norm: drop the radial grad
                    with torch.no_grad():
                        Wn = s.W_dec / s.W_dec.norm(dim=1, keepdim=True)
                        s.W_dec.grad -= (s.W_dec.grad * Wn).sum(1, keepdim=True) * Wn
                torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
                opts[k].step()
                with torch.no_grad():
                    if s.kind == "topk":
                        s.W_dec.data /= s.W_dec.data.norm(dim=1, keepdim=True)
                    last[k][parts.pop("fired")] = step
                rec[k] = parts
            step += 1
            if log and (step % 100 == 0 or step == 1):
                for k in saes:
                    rec[k] = {a: float(b) for a, b in rec[k].items()}
                    rec[k]["dead_frac"] = float(((step - 1 - last[k]) > dead_steps).float().mean())
                rec["elapsed_s"] = round(time.time() - t0, 1)
                hist.append(rec)
                log.write(json.dumps(rec) + "\n"); log.flush()
            out_of_time = time_limit and time.time() - t0 > time_limit
            if ckpt and (step % ckpt_every == 0 or step == total_steps or out_of_time):
                torch.save({"sae": {k: s.state_dict() for k, s in saes.items()},
                            "opt": {k: o.state_dict() for k, o in opts.items()},
                            "last": {k: v.cpu() for k, v in last.items()}, "step": step, "chunk": c + 1}, ckpt)
            if out_of_time:
                return step
        if step >= total_steps:
            break
    return step


@torch.no_grad()
def estimate_scale(A):
    """sqrt(E|x|^2 / d): dividing by it makes the mean squared norm equal d."""
    return float((A.pow(2).sum(-1).mean() / A.shape[-1]).sqrt())


class TopK:
    """Running top-k activations per latent over a stream of batches. Positions are global token indices."""
    def __init__(self, n, k, device="cuda"):
        self.v = torch.full((n, k), -1.0, device=device)
        self.i = torch.full((n, k), -1, dtype=torch.long, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.k = k

    def update(self, f, offset):
        """f [T, n] activations, offset = global index of row 0."""
        self.count += (f > 0).sum(0)
        k = min(self.k, f.shape[0])
        v, i = f.t().topk(k, dim=1)
        V = torch.cat([self.v, v], 1); I = torch.cat([self.i, i + offset], 1)
        top = V.topk(self.k, dim=1)
        self.v, self.i = top.values, I.gather(1, top.indices)


@torch.no_grad()
def evaluate_gpt2(model, layer, tokens, encode_decode, mean_act=None, sub=32, extra=None, baselines=True):
    """Splice x_hat = encode_decode(x) -> (x_hat, f) in at blocks.{layer}.hook_resid_pre for every position
    but BOS. Returns per-window arrays: squared error sum, total squared deviation from mean_act,
    L0 sum, token count, and summed next-token loss clean / spliced / zero / mean-ablated."""
    name = f"blocks.{layer}.hook_resid_pre"
    out = {k: [] for k in ["sq_err", "sq_dev", "l0", "n", "ce_clean", "ce_sae", "ce_zero", "ce_mean"]}
    store = {}

    def splice(act, hook):
        x = act[:, 1:]
        xh, f = encode_decode(x.reshape(-1, x.shape[-1]))
        xh = xh.view_as(x).to(act.dtype)
        store["sq_err"] = (x - xh).pow(2).sum(-1).sum(-1)
        store["sq_dev"] = (x - mean_act).pow(2).sum(-1).sum(-1)
        store["l0"] = (f > 0).sum(-1).view(x.shape[0], -1).sum(-1).float()
        if extra:
            extra(f, x)
        act[:, 1:] = xh
        return act

    def setv(v):
        def h(act, hook):
            act[:, 1:] = v
            return act
        return h

    def ce(logits, t):
        return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), t[:, 1:].flatten(), reduction="none").view(t.shape[0], -1).sum(-1)

    for j in range(0, len(tokens), sub):
        t = tokens[j:j + sub].to(torch.int64).cuda()
        if baselines:
            out["ce_clean"].append(ce(model(t), t))
        out["ce_sae"].append(ce(model.run_with_hooks(t, fwd_hooks=[(name, splice)]), t))
        for k in ["sq_err", "sq_dev", "l0"]:
            out[k].append(store[k])
        out["n"].append(torch.full((t.shape[0],), t.shape[1] - 1.0, device="cuda"))
        if baselines:
            out["ce_zero"].append(ce(model.run_with_hooks(t, fwd_hooks=[(name, setv(0.0))]), t))
            out["ce_mean"].append(ce(model.run_with_hooks(t, fwd_hooks=[(name, setv(mean_act))]), t))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items() if v}


def doc_bootstrap(per_window, doc, stat, n_boot=2000, seed=0):
    """95% CI of stat(dict of per-doc sums) resampling documents. per_window: dict name -> array."""
    docs = np.unique(doc)
    sums = {k: np.bincount(np.searchsorted(docs, doc), weights=v, minlength=len(docs)) for k, v in per_window.items()}
    rng = np.random.default_rng(seed)
    est = stat(sums)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(docs), len(docs))
        bs.append(stat({k: v[idx] for k, v in sums.items()}))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------- Gemma 2 + Gemma Scope ----------------

GEMMA_PATH = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16"


def load_gemma_scope(layer=12, width="16k", l0=82, repo="google/gemma-scope-2b-pt-res", device="cuda"):
    """Gemma Scope params.npz -> JumpReLU SAE (fp32). Downloads with huggingface_hub if not cached."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz")
    p = np.load(path)
    d, n = p["W_enc"].shape
    s = SAE(d, n, kind="jumprelu")
    with torch.no_grad():
        s.W_enc.copy_(torch.from_numpy(p["W_enc"])); s.W_dec.copy_(torch.from_numpy(p["W_dec"]))
        s.b_enc.copy_(torch.from_numpy(p["b_enc"])); s.b_dec.copy_(torch.from_numpy(p["b_dec"]))
        s.threshold.copy_(torch.from_numpy(p["threshold"]))
    s.source = path
    return s.to(device).eval()


class _Stop(Exception):
    pass


class GemmaRunner:
    """Gemma 2 in bf16 on the GPU via plain transformers (no TransformerLens weight processing).
    resid(tokens, layer): output of decoder layer `layer` (= TransformerLens blocks.{layer}.hook_resid_post),
    stopping the forward pass there. With `edit` set, a hook rewrites that output during a full forward pass."""
    def __init__(self, path=GEMMA_PATH, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        self.edit = None
        self._grab = None
        self._hooks = {}

    def _hook(self, layer):
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self._grab is not None:
                self._grab.append(hs)
                raise _Stop()
            if self.edit is not None:
                hs2 = self.edit(hs)
                return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
        return h

    def attach(self, layer):
        if layer not in self._hooks:
            self._hooks[layer] = self.model.model.layers[layer].register_forward_hook(self._hook(layer))

    @torch.no_grad()
    def resid(self, tokens, layer):
        self.attach(layer)
        self._grab = []
        try:
            self.model(input_ids=tokens, use_cache=False)
        except _Stop:
            pass
        h = self._grab[0]
        self._grab = None
        return h

    @torch.no_grad()
    def logits(self, tokens):
        return self.model(input_ids=tokens, use_cache=False).logits
