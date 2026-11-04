"""Throughput and FLOPs of the tiny U-Net and tiny DiT (bf16 autocast, RTX 4060 Laptop). Writes runs/timing.json."""
import json, sys, time, statistics
import torch
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import flow

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
for name, ctor in [("unet", flow.UNet), ("dit", flow.DiT)]:
    torch.manual_seed(0)
    net = ctor().to(dev)
    r = {"params": flow.n_params(net)}
    with FlopCounterMode(display=False) as fc:
        net(torch.randn(1, 1, 28, 28, device=dev), torch.rand(1, device=dev))
    r["gflops_fwd_per_image"] = fc.get_total_flops() / 1e9
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    x1 = torch.randn(128, 1, 28, 28, device=dev)

    def train_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow.flow_loss(net, x1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for _ in range(30): train_step()
    ts = []
    for _ in range(7):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(40): train_step()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 40)
    r["train_step_ms_b128"] = statistics.median(ts) * 1e3
    r["train_img_per_s"] = 128 / statistics.median(ts)
    net.eval()
    for bs, reps in [(1000, 10), (1, 50)]:
        x = torch.randn(bs, 1, 28, 28, device=dev); t = torch.rand(bs, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5): net(x, t)
            ts = []
            for _ in range(7):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(reps): net(x, t)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / reps)
        r[f"fwd_ms_b{bs}"] = statistics.median(ts) * 1e3
        r[f"fwd_img_per_s_b{bs}"] = bs / statistics.median(ts)
    r["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open("runs/timing.json", "w"), indent=1)
