# Part 6 benchmark: time full optimizer steps of the series' 88M model under one configuration.
# Imports model.py and data.py unchanged; attention variants are patched in this process only.
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import model as M                      # noqa: E402
from data import ShardLoader           # noqa: E402

T = 1024
FLOPS_PER_TOKEN = 622_950_000          # Part 1: 6N + 12*L*d*T for the 88M model (PaLM appendix B)


# ---------------------------------------------------------------- attention variants
_masks = {}


def _qkv(self, x, cos, sin):
    B, T_, _ = x.shape
    q = self.wq(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
    k = self.wk(x).view(B, T_, self.n_kv_head, self.hd).transpose(1, 2)
    v = self.wv(x).view(B, T_, self.n_kv_head, self.hd).transpose(1, 2)
    return M.apply_rope(q, cos, sin), M.apply_rope(k, cos, sin), v


def naive_forward(self, x, cos, sin):
    """Textbook attention: full T x T scores, a boolean causal mask, softmax in FP32."""
    B, T_, _ = x.shape
    q, k, v = _qkv(self, x, cos, sin)
    g = self.n_head // self.n_kv_head
    k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    if (T_, x.device) not in _masks:
        _masks[(T_, x.device)] = torch.ones(T_, T_, dtype=torch.bool, device=x.device).triu(1)
    s = (q @ k.transpose(-2, -1)) * self.hd ** -0.5
    s = s.masked_fill(_masks[(T_, x.device)], float("-inf"))
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return self.wo((p @ v).transpose(1, 2).reshape(B, T_, -1))


def expand_forward(self, x, cos, sin):
    """SDPA with K and V copied to every query head (for backends that refuse enable_gqa)."""
    B, T_, _ = x.shape
    q, k, v = _qkv(self, x, cos, sin)
    g = self.n_head // self.n_kv_head
    k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(y.transpose(1, 2).reshape(B, T_, -1))


def backend_ctx(name):
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    table = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
             "math": SDPBackend.MATH, "cudnn": SDPBackend.CUDNN_ATTENTION}
    return sdpa_kernel(table[name]) if name in table else nullcontext()


# ---------------------------------------------------------------- setup
def build(a):
    torch.manual_seed(a.seed)
    if a.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:                                               # tf32 and bf16 runs keep TF32 on
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.attn == "naive":
        M.Attention.forward = naive_forward
    elif a.gqa == "expand":
        M.Attention.forward = expand_forward
    model = M.GPT(M.MAIN).cuda()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": dict(fused=True), "foreach": dict(foreach=True), "forloop": dict(foreach=False)}[a.opt]
    opt = torch.optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, **kw)
    run = model
    if a.compile != "none":
        run = torch.compile(model, mode=None if a.compile == "default" else a.compile)
    return model, run, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="bf16")
    ap.add_argument("--attn", default="sdpa", help="sdpa (auto), flash, efficient, math, cudnn, naive")
    ap.add_argument("--gqa", choices=["native", "expand"], default="native")
    ap.add_argument("--compile", default="none", help="none, default, reduce-overhead, max-autotune, "
                                                      "max-autotune-no-cudagraphs")
    ap.add_argument("--opt", choices=["fused", "foreach", "forloop"], default="foreach")
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--tokens_per_step", type=int, default=2 ** 19)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=4, help="warm-up micro-batches before one optimizer step")
    ap.add_argument("--pin", type=int, default=0)
    ap.add_argument("--nb", type=int, default=0, help="non_blocking host-to-device copies")
    ap.add_argument("--micro_only", type=int, default=0, help="time N fwd+bwd micro-batches, no steps")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    model, run, opt = build(a)
    accum = a.tokens_per_step // (a.mb * T)
    loader = ShardLoader("train", B=a.mb, T=T)
    batches = loader.batches(step=0, accum=accum)
    data_s = [0.0]

    def get_batch():
        t = time.perf_counter()
        x, y = next(batches)
        if a.pin:
            x, y = x.pin_memory(), y.pin_memory()
        x, y = x.cuda(non_blocking=bool(a.nb)), y.cuda(non_blocking=bool(a.nb))
        data_s[0] += time.perf_counter() - t
        return x, y

    def micro(loss_sum):
        x, y = get_batch()
        if a.compile in ("reduce-overhead", "max-autotune"):
            torch.compiler.cudagraph_mark_step_begin()      # each micro-batch is a new CUDA graph replay
        if a.precision == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run(x, y)
        else:
            loss = run(x, y)
        (loss / accum).backward()
        return loss_sum + loss.detach()

    graphs = a.compile in ("reduce-overhead", "max-autotune")
    if graphs:                                          # CUDA graphs: .grad must not alias a graph's output buffer,
        for p in model.parameters():                    # so keep real gradient tensors and accumulate into them
            p.grad = torch.zeros_like(p)

    def opt_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=not graphs)

    res = dict(name=a.name, precision=a.precision, attn=a.attn, gqa=a.gqa, compile=a.compile, opt=a.opt,
               mb=a.mb, accum=accum, tokens_per_step=a.mb * T * accum, pin=a.pin, nb=a.nb,
               torch=torch.__version__)
    with backend_ctx(a.attn):
        # warm-up: compile (first call), CUDA graph recording, AdamW state allocation
        t0 = time.perf_counter()
        ls = torch.zeros((), device="cuda")
        first = None
        for i in range(a.warm):
            ls = micro(ls)
            if i == 0:
                torch.cuda.synchronize()
                first = time.perf_counter() - t0
        opt_step()
        torch.cuda.synchronize()
        res.update(first_micro_s=round(first, 2), warmup_s=round(time.perf_counter() - t0, 2))
        torch.cuda.reset_peak_memory_stats()

        if a.micro_only:
            times = []
            for i in range(a.micro_only):
                torch.cuda.synchronize(); t = time.perf_counter()
                ls = micro(ls); torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            med = sorted(times)[len(times) // 2]
            res.update(micro_s=[round(t, 4) for t in times], micro_s_median=med, tok_s=a.mb * T / med)
        else:
            times, opt_ms, first_ms, rest_ms = [], [], [], []
            for s in range(a.steps):
                data_s[0] = 0.0
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(accum + 2)]
                torch.cuda.synchronize(); t = time.perf_counter()
                ls = torch.zeros((), device="cuda")
                ev[0].record()
                for i in range(accum):
                    ls = micro(ls)
                    ev[i + 1].record()
                opt_step()
                ev[accum + 1].record()
                loss = (ls / accum).item()          # the one host sync per step, as a logger would do
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                times.append(dt)
                mic = [ev[i].elapsed_time(ev[i + 1]) for i in range(accum)]
                opt_ms.append(ev[accum].elapsed_time(ev[accum + 1]))
                first_ms.append(mic[0])
                rest = sorted(mic[1:]) if accum > 1 else mic
                rest_ms.append(rest[len(rest) // 2])
                print(json.dumps(dict(step=s, s=round(dt, 3), loss=round(loss, 4),
                                      tok_s=round(res["tokens_per_step"] / dt), opt_ms=round(opt_ms[-1], 2),
                                      first_micro_ms=round(mic[0], 1), micro_ms_median=round(rest_ms[-1], 1),
                                      data_ms=round(data_s[0] * 1000, 1), t=time.time())), flush=True)
            med = sorted(times)[len(times) // 2]
            res.update(step_s=[round(x, 3) for x in times], step_s_median=med,
                       tok_s=res["tokens_per_step"] / med, opt_ms_median=sorted(opt_ms)[len(opt_ms) // 2],
                       first_micro_ms=first_ms, micro_ms_median=sorted(rest_ms)[len(rest_ms) // 2])
    res["tflops"] = res["tok_s"] * FLOPS_PER_TOKEN / 1e12
    res["mfu"] = res["tflops"] / 25.3
    res["peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2 ** 30
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
