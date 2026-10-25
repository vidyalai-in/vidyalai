# Forward+backward time and peak memory at one micro-batch size, timed as in Part 1 (proto.py --micro_only).
import argparse, json, sys, time, torch
from common import heldout_batch
from model import GPT, MAIN, SMALL

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="main"); ap.add_argument("--mb", type=int, default=16)
ap.add_argument("--compile", type=int, default=0); ap.add_argument("--iters", type=int, default=14)
ap.add_argument("--impl", default="ours")   # ours = fromscratch/model.py, proto = Part 1's t01/proto.py
a = ap.parse_args()
torch.manual_seed(1337)
cfg = {"main": MAIN, "small": SMALL}[a.size]
if a.impl == "proto":
    sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/t01")
    import proto
    model = proto.GPT(*proto.SIZES[{"main": "t85", "small": "t20"}[a.size]]).cuda()
    model.num_params = lambda: sum(p.numel() for p in model.parameters())
else:
    model = GPT(cfg).cuda()
step_model = torch.compile(model) if a.compile else model
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
x, y = heldout_batch(a.mb, 1024)
accum = 2**19 // (a.mb * 1024)

def micro():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = step_model(x, y)
    (loss / accum).backward()

res = dict(impl=a.impl, size=a.size, mb=a.mb, compile=a.compile, N=model.num_params())
try:
    t0 = time.time()
    for _ in range(2): micro()                     # warm-up (compiles), then one optimizer step so AdamW states exist
    opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    res["warmup_s"] = round(time.time() - t0, 1)
    torch.cuda.reset_peak_memory_stats()
    times = []; res["t_start"] = time.time()
    for _ in range(a.iters):
        torch.cuda.synchronize(); t = time.perf_counter(); micro(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    res["t_end"] = time.time()
    ts = sorted(times[2:]); med = ts[len(ts) // 2]
    res.update(micro_s_median=round(med, 4), micro_s_min=round(ts[0], 4), micro_s_max=round(ts[-1], 4),
               tok_s=round(a.mb * 1024 / med), peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
               reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 3))
except torch.OutOfMemoryError as e:
    res.update(oom=True, peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3), msg=str(e)[:120])
print("RESULT " + json.dumps(res), flush=True)
