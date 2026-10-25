# Part 6: torch.profiler kernel table for one configuration (n micro-batches + one optimizer step).
import argparse, collections, json, os, sys, time
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                            # noqa: E402
from data import ShardLoader                            # noqa: E402

CATS = [  # first match wins
    ("optimizer", ("multi_tensor_apply", "adam", "Adam", "fused_adam")),
    ("attention", ("flash", "fmha", "efficient_attention", "attention_kernel", "cudnn", "sdpa", "softmax_warp")),
    ("matmul", ("gemm", "cutlass", "xmma", "ampere_", "sm80_", "sm86_", "sm89_", "sm90_", "cublas", "Kernel2",
                "splitKreduce", "s1688", "s16816", "h16816")),
    ("loss", ("log_softmax", "nll_loss", "cross_entropy", "LogSoftMax")),
]


def category(name):
    for cat, keys in CATS:
        if any(k in name for k in keys):
            return cat
    return "elementwise/norm"


def main():
    ap = argparse.ArgumentParser()
    for k, v in [("--precision", "bf16"), ("--attn", "sdpa"), ("--gqa", "native"), ("--compile", "none"),
                 ("--opt", "foreach"), ("--name", "prof")]:
        ap.add_argument(k, default=v)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pin", type=int, default=0)
    ap.add_argument("--nb", type=int, default=0)
    a = ap.parse_args()
    model, run, opt = bench.build(a)
    loader = ShardLoader("train", B=a.mb, T=1024)
    it = loader.batches(0, accum=1)

    def micro():
        x, y = next(it)
        if a.pin:
            x, y = x.pin_memory(), y.pin_memory()
        x, y = x.cuda(non_blocking=bool(a.nb)), y.cuda(non_blocking=bool(a.nb))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
            loss = run(x, y)
        loss.backward()

    def step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    with bench.backend_ctx(a.attn):
        for _ in range(a.warm):
            micro()
        step(); torch.cuda.synchronize()
        for _ in range(2):                              # second warm step: grads accumulate path
            micro(); micro(); step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(a.n):
                micro()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
    path = f"/home/amit/.cache/vidyalai-scratch/t06/logs/{a.name}.trace.json"
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ts0 = min(e["ts"] for e in kern)
    by_cat = collections.defaultdict(lambda: [0, 0.0])
    by_name = collections.defaultdict(lambda: [0, 0.0, ""])
    for e in kern:
        cat = category(e["name"]) if e["cat"] == "kernel" else e["cat"]
        by_cat[cat][0] += 1
        by_cat[cat][1] += e["dur"]
        by_name[e["name"]][0] += 1
        by_name[e["name"]][1] += e["dur"]
        by_name[e["name"]][2] = cat
    total = sum(v[1] for v in by_cat.values())
    ks = sorted(kern, key=lambda e: e["ts"])
    gaps, end = [], ks[0]["ts"] + ks[0]["dur"]
    for e in ks[1:]:
        if e["ts"] > end:
            gaps.append(e["ts"] - end)
        end = max(end, e["ts"] + e["dur"])
    big = sorted(gaps, reverse=True)[:8]
    span = (max(e["ts"] + e["dur"] for e in kern) - ts0)
    out = dict(name=a.name, mb=a.mb, n=a.n, compile=a.compile, attn=a.attn, precision=a.precision, opt=a.opt,
               wall_micro_ms=(t1 - t0) * 1000, wall_step_ms=(t2 - t1) * 1000, kernel_ms=total / 1000,
               gpu_span_ms=span / 1000, n_kernels=len(kern),
               idle_ms=sum(gaps) / 1000, n_gaps=len(gaps), biggest_gaps_us=[round(g) for g in big],
               cats={k: dict(n=v[0], ms=round(v[1] / 1000, 2), share=round(v[1] / total, 4))
                     for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1][1])},
               top=[dict(name=k[:160], cat=v[2], n=v[0], ms=round(v[1] / 1000, 2))
                    for k, v in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:40]])
    print("PROF " + json.dumps(out), flush=True)
    os.remove(path)


if __name__ == "__main__":
    main()
