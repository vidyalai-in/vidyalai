"""Score the tiny video models (and real clips) on per-frame quality and temporal consistency.
python eval_video.py --modes frames,full,factorized --seeds 3 --n 500 --steps 32"""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V, fmnist_eval as fe

ap = argparse.ArgumentParser(); ap.add_argument("--modes", default="real,shuffled,frames,full,factorized")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n", type=int, default=500)
ap.add_argument("--steps", type=int, default=32); ap.add_argument("--bs", type=int, default=125)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
Xt, Yt = fe.load("test"); Xt = Xt.to(dev)

def summarize(m, x):
    sc = fe.score(m["crops"])
    return {"fd": sc["fd"], "same": m["same"].tolist(), "agree": m["agree"].tolist(), "cos0": m["cos0"].tolist(),
            "cosn": m["cosn"].tolist(), "speed": m["speed"].tolist(), "accel": m["accel"].tolist(),
            "class_hist": sc["class_hist"], "mean_conf": sc["mean_conf"]}

results = {}
for mode in a.modes.split(","):
    per_seed, keep = [], None
    if mode not in ("real", "shuffled"):
        ck = torch.load(f"ckpt/{mode}.pt", map_location=dev)
        net = V.VideoDiT("factorized" if mode == "inflated" else mode).to(dev); net.load_state_dict(ck["ema"]); net.eval()
        # time one network call on a batch of bs clips (median of 7 after 3 warm-up)
        z = torch.randn(a.bs, 8, 1, 48, 48, device=dev)
        def call():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                net(z, torch.full((a.bs,), 0.5, device=dev))
        if dev == "cuda":
            with V.GpuMonitor() as mon:
                med, lo, hi = V.cuda_median(call, n=7, warmup=3)
        else:
            class _M: clock = power = 0.0
            mon, med, lo, hi = _M(), 0.0, 0.0, 0.0
        timing = {"call_batch": a.bs, "call_median_s": med, "call_min_s": lo, "call_max_s": hi,
                  "sm_clock_mhz": mon.clock, "power_w": mon.power, "train_steps": ck["step"], "train_time_s": ck["train_time"],
                  "params": ck["params"]}
    for s in range(a.seeds):
        g = torch.Generator(device=dev).manual_seed(500 + s)
        if mode in ("real", "shuffled"):
            x = V.moving_fmnist(Xt, a.n, generator=g)["x"]
            if mode == "shuffled":  # every frame from a different real clip: what 'no temporal link' looks like
                perm = torch.stack([torch.randperm(a.n, device=dev, generator=g) for _ in range(8)], 1)
                x = x[perm, torch.arange(8, device=dev)[None]]
        else:
            xs = []
            for i in range(0, a.n, a.bs):
                z = torch.randn(min(a.bs, a.n - i), 8, 1, 48, 48, device=dev, generator=g)
                xs.append(V.euler_sample(net, z, a.steps))
            x = torch.cat(xs)
        m = V.clip_metrics(x)
        per_seed.append(summarize(m, x))
        if s == 0:
            keep = ((x[:16].clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
        print(mode, s, f"fd {per_seed[-1]['fd']:.1f} same {np.mean(per_seed[-1]['same']):.3f} "
              f"cos0 {np.mean(per_seed[-1]['cos0']):.3f} speed {np.mean(per_seed[-1]['speed']):.2f} "
              f"accel {np.mean(per_seed[-1]['accel']):.2f}", flush=True)
    np.save(f"runs/clips-{mode}{'' if a.n >= 100 else '-test'}.npy", keep)
    r = {"per_seed": per_seed}
    if mode not in ("real", "shuffled"):
        r["timing"] = timing
    results[mode] = r
    json.dump(results, open(f"runs/exp2_eval_{a.modes.replace(',', '_')}_n{a.n}.json", "w"))
