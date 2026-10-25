# Throwaway Llama-style prototype for Part 1's throughput measurements.
# Not the series model (Part 4 writes that). Random tokens: throughput doesn't depend on the text.
import argparse, json, math, time, torch, torch.nn as nn, torch.nn.functional as F

SIZES = {  # name: (d_model, n_layer, n_head, n_kv_head, d_ff), head size 64
    "t20":  (384, 6, 6, 2, 1024),
    "t50":  (576, 8, 9, 3, 1536),
    "t85":  (768, 10, 12, 4, 2048),
    "t125": (768, 16, 12, 4, 2048),
}
V, T, HD = 32768, 1024, 64

def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, hd, 2).float() / hd)
    ang = torch.outer(torch.arange(T).float(), inv)          # (T, hd/2)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):                                  # x: (B, H, T, hd)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)

class Block(nn.Module):
    def __init__(s, d, nh, nkv, ff):
        super().__init__()
        s.nh, s.nkv = nh, nkv
        s.n1, s.n2 = nn.RMSNorm(d, eps=1e-5), nn.RMSNorm(d, eps=1e-5)
        s.wq = nn.Linear(d, nh * HD, bias=False)
        s.wkv = nn.Linear(d, 2 * nkv * HD, bias=False)
        s.wo = nn.Linear(nh * HD, d, bias=False)
        s.w1, s.w3 = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        s.w2 = nn.Linear(ff, d, bias=False)
    def forward(s, x, cos, sin):
        B, T, _ = x.shape
        h = s.n1(x)
        q = s.wq(h).view(B, T, s.nh, HD).transpose(1, 2)
        k, v = s.wkv(h).view(B, T, 2, s.nkv, HD).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + s.wo(a.transpose(1, 2).reshape(B, T, -1))
        h = s.n2(x)
        return x + s.w2(F.silu(s.w1(h)) * s.w3(h))

class GPT(nn.Module):
    def __init__(s, d, L, nh, nkv, ff):
        super().__init__()
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList(Block(d, nh, nkv, ff) for _ in range(L))
        s.norm = nn.RMSNorm(d, eps=1e-5)
        cos, sin = rope_cache(T, HD)
        s.register_buffer("cos", cos, persistent=False)
        s.register_buffer("sin", sin, persistent=False)
        for p in s.parameters():
            if p.dim() == 2: nn.init.normal_(p, std=0.02)
    def forward(s, idx, targets):
        x = s.emb(idx)
        for b in s.blocks: x = b(x, s.cos, s.sin)
        logits = F.linear(s.norm(x), s.emb.weight)            # tied output layer
        return F.cross_entropy(logits.float().view(-1, V), targets.reshape(-1))

def flops_per_token(model, d, L):
    N = sum(p.numel() for p in model.parameters())           # tied table counted once: it's the output matmul
    return 6 * N + 12 * L * d * T                              # PaLM appendix B convention

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="t85"); ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--tokens_per_step", type=int, default=2**19)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--micro_only", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1); ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    d, L, nh, nkv, ff = SIZES[a.size]
    model = GPT(d, L, nh, nkv, ff).cuda()
    N = sum(p.numel() for p in model.parameters())
    fpt = flops_per_token(model, d, L)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    step_model = torch.compile(model) if a.compile else model
    accum = a.tokens_per_step // (a.mb * T)
    def batch():
        x = torch.randint(0, V, (a.mb, T + 1), device="cuda")
        return x[:, :-1], x[:, 1:]
    def micro():
        x, y = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = step_model(x, y)
        (loss / accum).backward()
        return loss
    def opt_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    # warm-up: compile, plus one optimizer step so AdamW states exist
    t0 = time.time()
    for _ in range(2): micro()
    opt_step(); torch.cuda.synchronize()
    warm = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    res = dict(size=a.size, N=N, nonemb=N - V * d, d=d, L=L, mb=a.mb, accum=accum,
               tokens_per_step=a.mb * T * accum, flops_per_token=fpt, warmup_s=round(warm, 1))
    times = []
    if a.micro_only:                       # micro-batch search: time forward+backward only
        for i in range(a.micro_only):
            torch.cuda.synchronize(); t = time.perf_counter(); micro(); torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        ts = sorted(times[2:]); med = ts[len(ts) // 2]
        res.update(micro_s_median=med, tok_s=a.mb * T / med)
    else:
        for s in range(a.steps):
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(accum): loss = micro()
            opt_step(); torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            print(json.dumps(dict(step=s, s=round(times[-1], 3), loss=round(loss.item(), 4),
                                  tok_s=round(res["tokens_per_step"] / times[-1]), t=time.time())), flush=True)
        med = sorted(times)[len(times) // 2]
        res.update(step_s=times, step_s_median=med, tok_s=res["tokens_per_step"] / med)
    res["tflops"] = res["tok_s"] * fpt / 1e12
    res["mfu_vs_25.3"] = res["tflops"] / 25.3
    res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    res["reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    print("RESULT " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
