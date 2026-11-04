"""Exp 1: zero-shot CIFAR-10 with CLIP ViT-B/32 and SigLIP B/16; prompt conditions; features
for the linear probe; modality-gap stats; encoder throughput."""
import json, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import clip_utils as cu

OUT = "/home/amit/.cache/vidyalai-scratch/v03"
MODELS = {"clip-b32": "openai/clip-vit-base-patch32", "siglip-b16": "google/siglip-base-patch16-224"}
which = sys.argv[1]
name = MODELS[which]

test_imgs, test_y = cu.cifar10("test")
train_imgs, train_y = cu.cifar10("train")
model, proc, kind = cu.load_encoder(name)
print(which, sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)

# ---- timing (image tower only), batch 256, median of 7 after 2 warm-ups
batch = test_imgs[:256]
px = proc(images=batch, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
smi = subprocess.Popen(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE, text=True)
times = []
with torch.no_grad():
    for r in range(9):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5):
            model.get_image_features(pixel_values=px)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 5
        if r >= 2: times.append(dt)
smi.terminate(); smi_out = smi.communicate()[0].strip().splitlines()
clk = [float(l.split(",")[0]) for l in smi_out if l.strip()]
pw = [float(l.split(",")[1]) for l in smi_out if l.strip()]
timing = {"batch": 256, "median_s": float(np.median(times)), "runs": len(times),
          "img_per_s": 256 / float(np.median(times)), "sm_clock_mhz_median": float(np.median(clk)),
          "power_w_median": float(np.median(pw)), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
print(timing, flush=True)

# ---- features
t0 = time.time()
fte = cu.encode_images(model, proc, test_imgs)
ftr = cu.encode_images(model, proc, train_imgs)
print("encoded", fte.shape, ftr.shape, time.time() - t0, flush=True)
torch.save({"test": fte.half(), "train": ftr.half(), "test_y": test_y, "train_y": train_y}, f"{OUT}/feats-{which}.pt")

# ---- prompt conditions
C = cu.CIFAR10_CLASSES
conds = {"bare": ["{}"], "photo": ["a photo of a {}."], "this-is": ["this is a photo of {}."],
         "ensemble18": cu.CIFAR10_TEMPLATES}
res = {"model": name, "n_test": len(test_y), "timing": timing, "conds": {}}
correct = {}
for k, tmpl in conds.items():
    W = cu.zeroshot_classifier(model, proc, kind, C, tmpl)
    pred = (fte @ W.t()).argmax(1).numpy()
    correct[k] = (pred == test_y).astype(int)
    p, lo, hi = cu.wilson(int(correct[k].sum()), len(test_y))
    per_class = [float(correct[k][test_y == c].mean()) for c in range(10)]
    res["conds"][k] = {"acc": p, "lo": lo, "hi": hi, "per_class": per_class, "n_templates": len(tmpl)}
    print(k, round(p, 4), round(lo, 4), round(hi, 4), flush=True)
res["paired"] = {f"{a}-vs-{b}": cu.paired_diff_ci(correct[a], correct[b])
                 for a, b in [("photo", "bare"), ("ensemble18", "photo"), ("ensemble18", "bare"), ("this-is", "photo")]}
for k, v in res["paired"].items(): print(k, v)
np.save(f"{OUT}/correct-{which}.npy", np.stack([correct[k] for k in conds]))

# ---- modality gap on CIFAR-10 test: image vs prompt-text embeddings
T = cu.encode_texts(model, proc, [t.format(c) for c in C for t in cu.CIFAR10_TEMPLATES], kind)
Ty = np.repeat(np.arange(10), len(cu.CIFAR10_TEMPLATES))
rng = np.random.default_rng(0)
i = rng.choice(len(fte), 2000, replace=False)
I = fte[i]; Iy = test_y[i]
s_it = (I @ T.t()).numpy()
match = s_it[Iy[:, None] == Ty[None, :]]
mism = s_it[Iy[:, None] != Ty[None, :]]
s_ii = (I @ I.t()).numpy()[np.triu_indices(len(I), 1)]
s_tt = (T @ T.t()).numpy()[np.triu_indices(len(T), 1)]
gap = (I.mean(0) - T.mean(0)).norm().item()
res["gap"] = {"img_txt_match_mean": float(match.mean()), "img_txt_mismatch_mean": float(mism.mean()),
              "img_img_mean": float(s_ii.mean()), "txt_txt_mean": float(s_tt.mean()),
              "centroid_dist": gap, "n_img": len(I), "n_txt": len(T),
              "img_txt_match_q": np.percentile(match, [5, 50, 95]).tolist(),
              "img_img_q": np.percentile(s_ii, [5, 50, 95]).tolist()}
if kind == "clip":
    res["logit_scale"] = float(model.logit_scale.exp())
else:
    res["logit_scale"] = float(model.logit_scale.exp()); res["logit_bias"] = float(model.logit_bias)
print(res["gap"], res.get("logit_scale"), res.get("logit_bias"))
json.dump(res, open(f"{OUT}/exp1-{which}.json", "w"), indent=1)
