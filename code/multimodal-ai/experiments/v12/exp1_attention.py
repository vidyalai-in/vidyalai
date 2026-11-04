"""Experiment 1: cost of one self-attention call over a video latent, full vs factorized.
Assumed compression (CogVideoX / HunyuanVideo / Wan style): causal 3D VAE with 4x in time (first frame kept),
8x8 in space, then 1x2x2 patches. So tokens = (1 + (F-1)/4) x (H/16) x (W/16).
One attention layer of 8 heads x 64 dims (width 512), bf16, PyTorch SDPA. Median of 7 timed calls after 3 warm-up."""
import json, sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V
from torch.nn.attention import SDPBackend
clips = [  # (label, frames, height, width)
    ("1 s, 256x256, 17 frames", 17, 256, 256),
    ("2 s, 512x512, 33 frames", 33, 512, 512),
    ("6 s, 720x480, 49 frames at 8 fps", 49, 480, 720),
    ("5 s, 832x480, 81 frames at 16 fps", 81, 480, 832),
    ("5 s, 1280x720, 81 frames at 16 fps", 81, 720, 1280),
    ("5 s, 1280x720, 129 frames at 24 fps", 129, 720, 1280),
]
out = []
with V.GpuMonitor() as mon:
    for label, F_, H, W in clips:
        T, h, w = 1 + (F_ - 1) // 4, H // 16, W // 16
        for mode in ("full", "factorized"):
            r = V.attention_bench(T, h, w, mode=mode)
            r.update(label=label, frames=F_, height=H, width=W, backend="default")
            out.append(r); print(json.dumps(r), flush=True)
        if T * h * w <= 20000:  # textbook attention: materializes the (tokens x tokens) score matrix per head
            try:
                r = V.attention_bench(T, h, w, mode="full", backend=SDPBackend.MATH, n=5, warmup=2)
                r.update(label=label, frames=F_, height=H, width=W, backend="math")
            except torch.OutOfMemoryError as e:
                r = {"label": label, "tokens": T * h * w, "mode": "full", "backend": "math", "oom": True}
                torch.cuda.empty_cache()
            out.append(r); print(json.dumps(r), flush=True)
res = {"rows": out, "sm_clock_mhz": mon.clock, "power_w": mon.power, "gpu_mem_max_mib": mon.mem_max,
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
json.dump(res, open("runs/exp1_attention.json", "w"), indent=1)
print("clock", mon.clock, "power", mon.power)
