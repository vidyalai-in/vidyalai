"""FD-FMNIST vs sampling steps for DDPM and DDIM, plus timing. Resumable: skips configs already in the JSONL.

python eval_fm.py --sched linear --minutes 11 --configs "ddim:10,20,50,100,250,1000;ddpm:10,20,50,100,250,1000" --n 2000
python eval_fm.py --sched linear --timing
"""
import argparse, json, os, subprocess, sys, threading, time
import numpy as np
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ddpm, fmnist_eval as fe

p = argparse.ArgumentParser()
p.add_argument("--sched", default="linear"); p.add_argument("--minutes", type=float, default=11)
p.add_argument("--configs", default=""); p.add_argument("--n", type=int, default=2000)
p.add_argument("--seeds", default="0"); p.add_argument("--timing", action="store_true")
p.add_argument("--save_samples", default=""); p.add_argument("--ckpt", default="ckpt.pt"); p.add_argument("--tag", default="")
a = p.parse_args()
dev = "cuda"
run = f"/home/amit/.cache/vidyalai-scratch/v05/runs/fm_{a.sched}"
c = torch.load(f"{run}/{a.ckpt}", map_location=dev, weights_only=False)
st = c["state"]; arch = st.get("arch", {"ch": 32, "nres": 2})
model = ddpm.UNet(**arch).to(dev); model.load_state_dict(c["ema"]); model.eval()
net = torch.compile(model)
sched = ddpm.Schedule(a.sched)
out = f"{run}/sweep{a.tag}.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r["method"], r["steps"], r["n"], r["seed"]))


class Smi(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.rows, self.stop = [], False
    def run(self):
        while not self.stop:
            q = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            try: self.rows.append([float(v) for v in q])
            except ValueError: pass
            time.sleep(0.5)


with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # compile warm-up
    x = torch.randn(1000, 1, 28, 28, device=dev); t = torch.full((1000,), 500, device=dev)
    for _ in range(3): net(x, t)
torch.cuda.synchronize()

if a.timing:
    smi = Smi(); smi.start(); res = {}
    for bs in (1, 64, 1000):
        x = torch.randn(bs, 1, 28, 28, device=dev); t = torch.full((bs,), 500, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5): net(x, t)
            ts = []
            for _ in range(7):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(20): net(x, t)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 20)
        res[f"fwd_bs{bs}_ms"] = [round(v * 1000, 3) for v in ts]
    # full sampling runs, batch 1000, median of 5
    for steps in (10, 50):
        ts = []
        for k in range(6):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ddpm.sample(net, (1000, 1, 28, 28), sched, steps=steps, method="ddim")
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        res[f"sample_ddim{steps}_bs1000_s"] = [round(v, 4) for v in ts[1:]]
    smi.stop = True; smi.join()
    s = np.array(smi.rows)
    res["smi_median"] = {"sm_mhz": float(np.median(s[:, 0])), "power_w": float(np.median(s[:, 1])), "temp_c": float(np.median(s[:, 2])), "n": len(s)}
    res["torch"] = torch.__version__; res["gpu"] = torch.cuda.get_device_name()
    json.dump(res, open(f"{run}/timing{a.tag}.json", "w"), indent=1); print(json.dumps(res, indent=1))
    sys.exit()

t_end = time.time() + a.minutes * 60
todo = []
for part in a.configs.split(";"):
    m, ss = part.split(":")
    for s_ in ss.split(","):
        for seed in a.seeds.split(","):
            todo.append((m, int(s_), a.n, int(seed)))
for m, steps, n, seed in todo:
    if (m, steps, n, seed) in done:
        continue
    est = steps * n / 1000 * 0.12
    if time.time() + est > t_end:
        print("out of time before", m, steps, n, seed); continue
    g = torch.Generator(device=dev).manual_seed(100 + seed)
    torch.cuda.synchronize(); t0 = time.time()
    xs = [ddpm.sample(net, (1000, 1, 28, 28), sched, steps=steps, method=m, generator=g).float().cpu() for _ in range(n // 1000)]
    torch.cuda.synchronize(); dt = time.time() - t0
    x = torch.cat(xs); r = fe.score(x)
    rec = {"sched": a.sched, "method": m, "steps": steps, "n": n, "seed": seed, "fd": r["fd"], "mean_conf": r["mean_conf"],
           "class_hist": r["class_hist"], "sample_s": dt, "train_step": st["step"]}
    open(out, "a").write(json.dumps(rec) + "\n")
    if a.save_samples:
        np.save(f"{run}/samples{a.tag}_{m}{steps}_s{seed}.npy", x[:64].numpy().astype(np.float16))
    print(f"{m} {steps} n={n} seed={seed} FD {r['fd']:.2f} ({dt:.0f}s)", flush=True)
