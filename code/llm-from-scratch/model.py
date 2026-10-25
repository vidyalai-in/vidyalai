"""The series' model: a Llama-style decoder in plain PyTorch (Build an LLM From Scratch, Part 4).

Token embedding (tied with the output layer), RMSNorm, rotary position embeddings (RoPE),
grouped-query attention, a SwiGLU MLP, pre-norm residual blocks, a final norm, logits and
the cross-entropy loss, GPT-2-style initialization, and a simple sampler.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    d_model: int = 768          # width of the residual stream
    n_layer: int = 10
    n_head: int = 12            # query heads
    n_kv_head: int = 4          # key/value heads; each is shared by n_head // n_kv_head query heads
    d_ff: int = 2048            # SwiGLU hidden width
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    @property
    def head_dim(self):
        return self.d_model // self.n_head


MAIN = GPTConfig()                                                            # 88,096,512 parameters
SMALL = GPTConfig(d_model=384, n_layer=6, n_head=6, n_kv_head=2, d_ff=1024)   # 22,025,088 parameters


def param_formula(cfg):
    """Closed-form parameter count, component by component (tied embeddings counted once)."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_head * hd + 2 * d * cfg.n_kv_head * hd + cfg.n_head * hd * d   # Q, K and V, output
    mlp = 3 * d * cfg.d_ff                                                        # gate, up, down
    norms = 2 * d                                                                 # two RMSNorm gains
    return {"embedding": cfg.vocab_size * d, "attention": cfg.n_layer * attn, "mlp": cfg.n_layer * mlp,
            "block norms": cfg.n_layer * norms, "final norm": d}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()                                       # normalize in FP32, even under BF16
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.type_as(x)


def rope_tables(cfg):
    """cos and sin of every angle, for each position (rows) and rotation frequency (columns)."""
    hd = cfg.head_dim
    inv_freq = cfg.rope_base ** (-torch.arange(0, hd, 2).float() / hd)   # radians per position: 1 down to 1/7499
    angles = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)  # (max_seq_len, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair (x[2i], x[2i+1]) of every head vector by angle position * inv_freq[i]."""
    T = x.shape[-2]                                           # x: (B, heads, T, head_dim)
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()      # the two halves of each pair
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)                         # interleave the pairs back


class Attention(nn.Module):
    """Causal grouped-query attention: n_head query heads share n_kv_head key/value heads."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv_head, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)       # (B, 12, T, 64)
        k = self.wk(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)    # (B, 4, T, 64)
        v = self.wv(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)    # (B, 4, T, 64)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))                    # heads side by side


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: each branch reads a normalized copy and adds its result back."""

    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg=MAIN):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)   # recomputed, never saved
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        """GPT-2's scheme, as in nanoGPT: N(0, 0.02) for the embedding and every linear layer,
        N(0, 0.02 / sqrt(2 * n_layer)) for the two projections per block that write into the
        residual stream, and gains of 1 for the norms."""
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.ones_(p)
            elif name.endswith(("attn.wo.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.cfg.n_layer))
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())     # the tied table is counted once

    def forward(self, idx, targets=None):
        """Logits (B, T, vocab) if no targets; otherwise only the mean next-token loss, so that a
        compiled training step never has to keep the full logits tensor around."""
        x = self.tok_emb(idx)                                  # (B, T) ids -> (B, T, d)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied output layer: (B, T, vocab)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Sample one token at a time (no KV cache: every step reruns the whole window)."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.max_seq_len:])[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)                    # greedy
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
