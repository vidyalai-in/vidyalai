# Exp 1a: latency, throughput, peak GPU memory and FLOPs vs input resolution, ViT-B/16 vs ResNet-50 (fp16).
import sys, json, time, torch, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vit as L
from torch.utils.flop_counter import FlopCounterMode
torch.backends.cudnn.benchmark = True
RES = [224, 320, 448, 640, 896, 1024]
B = 16
models = {"vit_b16": L.load_hf("google/vit-base-patch16-224"),
          "vit_b16_eager": L.load_hf("google/vit-base-patch16-224", attn="eager"),
          "resnet50": L.load_hf("microsoft/resnet-50")}
rows = []
for name, m in models.items():
    kw = {"interpolate_pos_encoding": True} if name.startswith("vit") else {}
    for r in RES:
        row = {"model": name, "res": r, "tokens": (r // 16) ** 2 + 1 if name.startswith("vit") else None}
        x1 = torch.randn(1, 3, r, r, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                m(pixel_values=x1, **kw)
            row["gflops_b1"] = fc.get_total_flops() / 1e9
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            m(pixel_values=x1, **kw); torch.cuda.synchronize()
            row["peak_act_mb_b1"] = (torch.cuda.max_memory_allocated() - base) / 2**20
            if name == "vit_b16_eager":
                rows.append(row); print(row, flush=True); continue
            with L.GpuSampler() as gs:
                lat = L.time_cuda(lambda: m(pixel_values=x1, **kw), warmup=5, reps=15)
                xb = torch.randn(B, 3, r, r, device="cuda", dtype=torch.float16)
                torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
                tb = L.time_cuda(lambda: m(pixel_values=xb, **kw), warmup=3, reps=7)
                row["peak_act_mb_b16"] = (torch.cuda.max_memory_allocated() - base) / 2**20
            row.update({"lat_ms_b1_median": float(np.median(lat)), "lat_ms_b1_all": lat,
                        "b16_ms_median": float(np.median(tb)), "b16_ms_all": tb,
                        "img_per_s_b16": B / (np.median(tb) / 1e3), **gs.summary()})
            del xb
        rows.append(row); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if not k.endswith("_all")}, flush=True)
meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "dtype": "float16", "batch": B,
        "params_vit": L.n_params(models["vit_b16"]), "params_resnet": L.n_params(models["resnet50"])}
json.dump({"meta": meta, "rows": rows}, open("/home/amit/.cache/vidyalai-scratch/v02/results/bench.json", "w"), indent=1)
print(meta)
