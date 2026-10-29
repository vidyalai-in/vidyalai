"""Training memory: where it goes, and the tools that shrink it (Build an LLM From Scratch, Part 14).

    python memory.py      # predicted memory per component for MAIN at micro-batch 16, BF16 autocast

state_bytes()        weights, gradients and optimizer state for a parameter count, per scheme
activation_bytes()   what one eager BF16 micro-batch of model.py saves for the backward pass
korthikanti()        the per-layer activation formula of Korthikanti et al. (2022), for comparison
zero_bytes()         model-state bytes per GPU under ZeRO stages 0 to 3 (Rajbhandari et al., 2020)
checkpoint_blocks()  recompute chosen blocks (all, every k-th, or selectively) during the backward pass
ChunkedLossGPT       the model with a loss that never holds the full (tokens x vocabulary) logits
add_lora()           freeze the model and train low-rank adapters on its linear layers instead
CPUAdamW             AdamW whose FP32 master weights and moments live in CPU memory
"""
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from model import GPT, MAIN, param_formula

GiB = 2 ** 30

# bytes per parameter: (weights, gradients, optimizer state)
SCHEMES = {
    "fp32-adamw": (4, 4, 8),    # train.py: FP32 weights cast to BF16 on the fly, FP32 gradients, two FP32 moments
    "zero-paper": (2, 2, 12),   # ZeRO's mixed precision: 16-bit weights and gradients; FP32 master copy and moments
    "adamw-8bit": (4, 4, 2),    # bitsandbytes AdamW8bit: both moments in 8 bits (plus one scale per 256 values)
    "pure-bf16": (2, 2, 4),     # weights, gradients and both moments in BF16, no FP32 copy anywhere
}


def state_bytes(n_params, scheme="fp32-adamw"):
    w, g, o = SCHEMES[scheme]
    return {"weights": w * n_params, "gradients": g * n_params, "optimizer": o * n_params}


def activation_bytes(cfg, tokens, compiled=False):
    """Bytes saved for the backward pass by one micro-batch of `tokens` tokens under BF16 autocast.

    Eager: the residual stream is FP32 (the embedding table is FP32), so each RMSNorm keeps its FP32 input
    and FP32 output, and every linear layer keeps its own BF16 cast of its input. SDPA's flash kernel keeps
    q, k, v, its output and one FP32 log-sum-exp per head and token, never the T x T scores.
    compiled=True: what torch.compile's graph keeps instead (read from a memory snapshot): one BF16 norm
    output per sublayer, no SiLU output, and BF16 logits rather than an FP32 log-softmax."""
    d, f, V = cfg.d_model, cfg.d_ff, cfg.vocab_size
    q, kv = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    flash = 2 * q + 2 * kv + 2 * kv + 4 * cfg.n_head + 2 * q   # q, k after RoPE, v, log-sum-exp, output
    if compiled:
        attn = 4 * d + 2 * d + flash + 2 * q     # FP32 residual, BF16 norm output, flash, wo's input
        mlp = 4 * d + 2 * d + 3 * 2 * f          # FP32 residual, BF16 norm output, gate, up, their product
        head = 4 * d + 2 * d + 2 * V             # FP32 final norm input, BF16 norm output, BF16 logits
    else:
        attn = (4 * d + 4 * d          # norm: FP32 input, FP32 output
                + 3 * 2 * d            # BF16 casts of the norm output, one each for wq, wk, wv
                + flash + 2 * q)       # wo's input, a contiguous copy of the flash output, heads side by side
        mlp = (4 * d + 4 * d           # norm: FP32 input, FP32 output
               + 2 * 2 * d             # BF16 casts for gate and up
               + 4 * 2 * f)            # gate output (for SiLU), SiLU output, up output, their product
        head = 4 * d + 4 * d + 2 * d + 4 * V   # final norm, its BF16 cast, the FP32 log-softmax of the logits
    return {"attention": cfg.n_layer * attn * tokens, "mlp": cfg.n_layer * mlp * tokens,
            "output layer and loss": head * tokens,
            "BF16 weight copies": 2 * sum(param_formula(cfg).values())}   # autocast's casts, kept for backward


def korthikanti(s, b, h, a):
    """Activation bytes per transformer layer, Korthikanti et al. (2022), Eq. 1: sbh(34 + 5as/h).
    Assumes a GPT-3-style layer: 16-bit activations, 1-byte dropout masks, a 4h GeLU MLP, LayerNorm,
    and attention that stores its a x s x s scores."""
    return s * b * h * (34 + 5 * a * s / h)


def zero_bytes(n_params, n_gpus, stage, k=12):
    """Model-state bytes per GPU: 2 for 16-bit weights, 2 for gradients, k for optimizer state (ZeRO's
    accounting; k = 12 for mixed-precision Adam). Stage 1 shards the optimizer state, 2 adds gradients,
    3 adds the weights."""
    w, g, o = 2 * n_params, 2 * n_params, k * n_params
    if stage >= 1:
        o /= n_gpus
    if stage >= 2:
        g /= n_gpus
    if stage >= 3:
        w /= n_gpus
    return w + g + o


# ---------------------------------------------------------------- activation checkpointing

_KEEP = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
         torch.ops.aten._scaled_dot_product_flash_attention.default,
         torch.ops.aten._scaled_dot_product_efficient_attention.default,
         torch.ops.aten._scaled_dot_product_cudnn_attention.default}


def _keep_matmuls(ctx, op, *args, **kwargs):
    """Selective policy: store what the matmuls and attention produce, recompute everything in between."""
    return CheckpointPolicy.MUST_SAVE if op in _KEEP else CheckpointPolicy.PREFER_RECOMPUTE


def _recompute(forward, context_fn, *args):
    if context_fn is None:
        return checkpoint(forward, *args, use_reentrant=False)
    return checkpoint(forward, *args, use_reentrant=False, context_fn=context_fn)


def checkpoint_blocks(model, every=1, selective=False):
    """Make every `every`-th block keep only its input and rerun its forward during the backward pass.
    selective=True keeps the matmul and attention outputs too and reruns only the cheap ops between them."""
    context_fn = functools.partial(create_selective_checkpoint_contexts, _keep_matmuls) if selective else None
    for i, block in enumerate(model.blocks):
        if i % every == 0:
            block.forward = functools.partial(_recompute, block.forward, context_fn)
    return model


def _chunk_loss(h, w, t):
    return F.cross_entropy(F.linear(h, w).float(), t, ignore_index=-1, reduction="sum")


class ChunkedLossGPT(GPT):
    """GPT whose training loss never builds the full logits: the final norm's output is cut into chunks of
    `chunk` tokens, and each chunk's logits are recomputed during the backward pass instead of stored."""
    chunk = 2048

    def forward(self, idx, targets=None):
        if targets is None:
            return super().forward(idx)
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        h, t = self.norm(x).flatten(0, 1), targets.reshape(-1)
        total = sum(checkpoint(_chunk_loss, h[i:i + self.chunk], self.tok_emb.weight, t[i:i + self.chunk],
                               use_reentrant=False) for i in range(0, t.numel(), self.chunk))
        return total / (t != -1).sum()


# ---------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: y = W x + (alpha / r) B A x, B starting at 0."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.a = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def add_lora(model, r=8, alpha=16, targets=("wq", "wk", "wv", "wo", "gate", "up", "down")):
    """Freeze every weight, then wrap the named linear layers with LoRA adapters (the only trainable part)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


# ---------------------------------------------------------------- optimizer offload

class CPUAdamW:
    """AdamW with the FP32 master weights and both moments in CPU memory (ZeRO-Offload's split, done by hand).

    The GPU keeps only the weights the model computes with (FP32 or BF16) and their gradients. step() copies
    the gradients to pinned CPU buffers, runs torch's AdamW on the CPU, and copies the new weights back
    through the same buffers. CPU memory: 12 bytes per parameter plus one buffer in the GPU's dtype."""

    def __init__(self, groups, **kw):
        self.gpu, cpu_groups = [], []
        for g in groups:
            ps = [p for p in g["params"] if p.requires_grad]
            self.gpu += ps
            masters = [p.detach().float().cpu().requires_grad_() for p in ps]   # FP32 copies in CPU memory
            cpu_groups.append({**g, "params": masters})
        self.cpu = [p for g in cpu_groups for p in g["params"]]
        self.bufs = [torch.empty(p.shape, dtype=q.dtype).pin_memory() for p, q in zip(self.cpu, self.gpu)]
        self.opt = torch.optim.AdamW(cpu_groups, **kw)
        self.param_groups = self.opt.param_groups

    @torch.no_grad()
    def step(self):
        for buf, p in zip(self.bufs, self.gpu):
            buf.copy_(p.grad, non_blocking=True)                 # gradient: GPU -> pinned CPU buffer
        torch.cuda.synchronize()
        for c, buf in zip(self.cpu, self.bufs):
            c.grad = buf.float()                                 # FP32 gradient for the FP32 update
        self.opt.step()
        for buf, c, p in zip(self.bufs, self.cpu, self.gpu):
            buf.copy_(c)                                         # new FP32 weights -> GPU dtype, same buffer
            p.copy_(buf, non_blocking=True)                      # pinned CPU -> GPU

    def zero_grad(self, set_to_none=True):
        for p in self.gpu:
            p.grad = None


if __name__ == "__main__":
    n, tokens = sum(param_formula(MAIN).values()), 16 * 1024
    for compiled in (False, True):
        rows = {**state_bytes(n), **activation_bytes(MAIN, tokens, compiled)}
        print("compiled" if compiled else "eager")
        for k, v in rows.items():
            print(f"  {k:24} {v / GiB:6.3f} GiB")
        print(f"  {'total':24} {sum(rows.values()) / GiB:6.3f} GiB")
    print(f"Korthikanti, per layer at s=1024, b=16, h=768, a=12: {korthikanti(1024, 16, 768, 12) / GiB:.3f} GiB")
