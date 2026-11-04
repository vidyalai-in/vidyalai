"""All Fashion-MNIST sampling measurements in priority order, in one process (one compile), resumable.

python sweep_all.py --minutes 11
Writes runs/sweep.jsonl (one line per config) and runs/timing.json.
"""
import argparse, json, os, subprocess, sys, threading, time
import numpy as np
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ddpm, fmnist_eval as fe

p = argparse.ArgumentParser(); p.add_argument("--minutes", type=float, default=11); a = p.parse_args()
dev = "cuda"; R = "/home/amit/.cache/vidyalai-scratch/v05/runs"
OUT = f"{R}/sweep.jsonl"; t_end = time.time() + a.minutes * 60
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r["model"], r["method"], r["steps"], r["n"], r["seed"]))

MODELS = {"linear": ("linear", f"{R}/fm_linear/ckpt.pt"), "linear_c1": ("linear", f"{R}/fm_linear/ckpt_c1.pt"),
          "cosine": ("cosine", f"{R}/fm_cosine/ckpt.pt")}
JOBS = []
for m in ("ddim", "ddpm"):
    for s in (10, 20, 50, 100, 250): JOBS.append(("linear", m, s, 2000, 0))
JOBS.append(("linear", "ddpm", 1000, 2000, 0))
JOBS.append(("TIMING",))
for mod in ("cosine",):
    for m in ("ddim", "ddpm"):
        for s in (10, 50, 250): JOBS.append((mod, m, s, 2000, 0))
JOBS += [("linear", "ddim", 10, 2000, 1), ("linear", "ddim", 50, 2000, 1), ("linear", "ddpm", 50, 2000, 1),
         ("linear", "ddim", 50, 5000, 0), ("linear", "ddim", 1000, 2000, 0), ("linear", "ddpm", 250, 2000, 1)]

net, cur = None, None
model = ddpm.UNet(ch=32, nres=2).to(dev).eval()


def load(name):
    global net, cur
    if cur == name: return
    c = torch.load(MODELS[name][1], map_location=dev, weights_only=False)
    assert c["state"]["arch"] == {"ch": 32, "nres": 2}
    model.load_state_dict(c["ema"]); cur = name
    load.step = c["state"]["step"]
    if net is None:
        net = torch.compile(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.randn(1000, 1, 28, 28, device=dev); t = torch.full((1000,), 500, device=dev)
            for _ in range(3): net(x, t)


class Smi(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.rows, self.stop = [], False
    def run(self):
        while not self.stop:
            q = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
            try: self.rows.append([float(v) for v in q])
            except ValueError: pass
            time.sleep(0.5)


def timing():
    load("linear"); smi = Smi(); smi.start(); res = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bs in (1, 100, 1000):
            x = torch.randn(bs, 1, 28, 28, device=dev); t = torch.full((bs,), 500, device=dev)
            for _ in range(5): net(x, t)
            ts = []
            for _ in range(7):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(20): net(x, t)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 20)
            res[f"fwd_bs{bs}_ms"] = [round(v * 1000, 3) for v in ts]
    sched = ddpm.Schedule("linear")
    for m, steps in (("ddim", 50), ("ddpm", 50)):
        ts = []
        for k in range(6):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ddpm.sample(net, (1000, 1, 28, 28), sched, steps=steps, method=m)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        res[f"sample_{m}{steps}_bs1000_s"] = [round(v, 4) for v in ts[1:]]
    smi.stop = True; smi.join(); s = np.array(smi.rows)
    res["smi_median"] = {"sm_mhz": float(np.median(s[:, 0])), "power_w": float(np.median(s[:, 1])),
                         "temp_c": float(np.median(s[:, 2])), "util": float(np.median(s[:, 3])), "n": len(s)}
    res["torch"] = torch.__version__; res["gpu"] = torch.cuda.get_device_name()
    json.dump(res, open(f"{R}/timing.json", "w"), indent=1); print(json.dumps(res), flush=True)


for job in JOBS:
    if job[0] == "TIMING":
        if not os.path.exists(f"{R}/timing.json") and time.time() + 60 < t_end: timing()
        continue
    mod, m, steps, n, seed = job
    if (mod, m, steps, n, seed) in done or not os.path.exists(MODELS[mod][1]): continue
    if time.time() + steps * n / 1000 * 0.112 + 12 > t_end:
        print("skip (time)", job, flush=True); continue
    load(mod); sched = ddpm.Schedule(MODELS[mod][0])
    g = torch.Generator(device=dev).manual_seed(100 + seed)
    torch.cuda.synchronize(); t0 = time.time()
    x = torch.cat([ddpm.sample(net, (1000, 1, 28, 28), sched, steps=steps, method=m, generator=g).float().cpu() for _ in range(n // 1000)])
    torch.cuda.synchronize(); dt = time.time() - t0
    r = fe.score(x)
    rec = {"model": mod, "sched": MODELS[mod][0], "train_step": load.step, "method": m, "steps": steps, "n": n, "seed": seed,
           "fd": r["fd"], "mean_conf": r["mean_conf"], "class_hist": r["class_hist"], "sample_s": dt}
    open(OUT, "a").write(json.dumps(rec) + "\n")
    np.save(f"{R}/samples_{mod}_{m}{steps}_s{seed}.npy", x[:64].numpy().astype(np.float16))
    print(f"{mod} {m} {steps} n={n} seed={seed} FD {r['fd']:.2f} ({dt:.0f}s)", flush=True)
print("sweep pass done", flush=True)
