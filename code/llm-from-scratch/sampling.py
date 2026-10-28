"""Batched sampling with a KV cache (Build an LLM From Scratch, Part 11).

    from sampling import generate_batch
    replies = generate_batch(model, [prompt_ids(m, tok) for m in convs], max_new=256, temperature=1.0)

Part 4's model.generate reruns the whole window for every new token, one conversation at a time.
generate_batch left-pads a batch of prompts, keeps every layer's keys and values, and feeds one new
token per step. Same weights and same math as model.forward: RoPE positions count from each
prompt's first real token, and padding is masked out of attention.

Memory: the cache holds 2 x layers x kv_heads x (prompt + reply) x head_dim BF16 numbers per row, about
10 MB at 1,024 tokens, and the prefill builds a (rows, P, P) mask. 64 rows run comfortably on the 8 GB
laptop GPU; 256 rows with prompts up to 512 tokens filled it (7.7 GB used) and slowed to a crawl.
"""
import torch
import torch.nn.functional as F

END = 4                                                    # Part 2's <|end|>


def _rope(x, cos, sin):
    """model.apply_rope with per-row positions: cos and sin are (B, 1, t, head_dim / 2)."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)


def _forward(model, idx, pos, cache, at, mask):
    """Run new tokens idx (B, t) at positions pos (B, t), writing their keys and values into the
    cache at columns at..at+t. mask (B, 1, t, at+t) says which keys each new token may read.
    Returns the logits of the last new token."""
    cos, sin = model.rope_cos[pos][:, None], model.rope_sin[pos][:, None]
    x = model.tok_emb(idx)
    B, t = idx.shape
    for blk, (K, V) in zip(model.blocks, cache):
        a, h = blk.attn, blk.attn_norm(x)
        q = _rope(a.wq(h).view(B, t, a.n_head, a.hd).transpose(1, 2), cos, sin)
        K[:, :, at:at + t] = _rope(a.wk(h).view(B, t, a.n_kv_head, a.hd).transpose(1, 2), cos, sin)
        V[:, :, at:at + t] = a.wv(h).view(B, t, a.n_kv_head, a.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, K[:, :, :at + t], V[:, :, :at + t], attn_mask=mask,
                                           enable_gqa=True)
        x = x + a.wo(y.transpose(1, 2).reshape(B, t, -1))
        x = x + blk.mlp(blk.mlp_norm(x))
    return F.linear(model.norm(x[:, -1]), model.tok_emb.weight).float()


@torch.no_grad()
def generate_batch(model, prompts, max_new=256, temperature=1.0, top_k=None, seed=0, eos_id=END):
    """Replies to a batch of prompts (lists of token ids). temperature=0 is greedy. Returns one
    (reply ids without the end token, ended) per prompt; a reply stops at its first eos_id."""
    cfg, B = model.cfg, len(prompts)
    P = max(len(p) for p in prompts)
    assert P + max_new <= cfg.max_seq_len, "prompt + reply must fit the RoPE table"
    idx = torch.zeros(B, P, dtype=torch.long)
    valid = torch.zeros(B, P + max_new, dtype=torch.bool)
    for r, p in enumerate(prompts):                        # left padding: every prompt ends at column P
        idx[r, P - len(p):] = torch.tensor(p)
        valid[r, P - len(p):P] = True
    idx, valid = idx.cuda(), valid.cuda()
    pos = (valid[:, :P].long().cumsum(1) - 1).clamp(min=0)  # position 0 = the prompt's first real token
    eye = torch.eye(P, dtype=torch.bool, device="cuda")     # a padding query reads itself, never NaN
    mask = (torch.tril(torch.ones(P, P, dtype=torch.bool, device="cuda")) & valid[:, None, :P]) | eye
    kv = (B, cfg.n_kv_head, P + max_new, cfg.head_dim)
    cache = [(torch.zeros(kv, dtype=torch.bfloat16, device="cuda"),
              torch.zeros(kv, dtype=torch.bfloat16, device="cuda")) for _ in model.blocks]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out, done, last = [], torch.zeros(B, dtype=torch.bool, device="cuda"), pos[:, -1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = _forward(model, idx, pos, cache, 0, mask[:, None])
        for s in range(max_new):
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]
            out.append(nxt)
            done |= nxt == eos_id
            if done.all() or s == max_new - 1:
                break
            valid[:, P + s] = True
            last = last + 1
            logits = _forward(model, nxt[:, None], last[:, None], cache, P + s,
                              valid[:, None, None, :P + s + 1])
    rows = torch.stack(out, 1).tolist()
    replies = []
    for row in rows:
        ended = eos_id in row
        replies.append((row[:row.index(eos_id)] if ended else row, ended))
    return replies
