# Exp 4: high-norm "artifact" tokens. Output-token norms of patch tokens in DINOv2-B/14 with and without registers,
# and in the supervised ViT-B/16, on 256 Pet test images at 224 px.
import sys, io, json, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vit as L
df = pd.read_parquet(L.pets_parquet("test")).iloc[::14].reset_index(drop=True)[:256]
def batch(ims, r, norm):
    xs = []
    for b in ims:
        x = torch.from_numpy(np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))).permute(2, 0, 1).cuda().float()[None] / 255
        h, w = x.shape[-2:]; s = r / min(h, w)
        x = F.interpolate(x, size=(max(r, round(h * s)), max(r, round(w * s))), mode="bicubic", antialias=True)
        h, w = x.shape[-2:]; t, l = (h - r) // 2, (w - r) // 2
        x = x[..., t:t + r, l:l + r].clamp(0, 1)
        m, sd = L.MEAN_STD[norm]
        xs.append((x - torch.tensor(m, device="cuda").view(1, 3, 1, 1)) / torch.tensor(sd, device="cuda").view(1, 3, 1, 1))
    return torch.cat(xs).half()
out = {}
for name, norm, nreg in [("facebook/dinov2-base", "imagenet", 0), ("facebook/dinov2-with-registers-base", "imagenet", 4),
                         ("google/vit-base-patch16-224", "vit", 0)]:
    m = L.load_hf(name)
    enc = m.vit if hasattr(m, "vit") else m
    per_layer = []; last = []; regn = []; clsn = []
    with torch.no_grad():
        for s in range(0, len(df), 32):
            x = batch(df["image"][s:s + 32], 224, norm)
            o = enc(pixel_values=x, output_hidden_states=True)
            hs = o.hidden_states  # embeddings + one per layer (pre final LayerNorm)
            pre = 1 + nreg
            per_layer.append(torch.stack([h[:, pre:].float().norm(dim=-1).flatten() for h in hs[1:]]).cpu())
            last.append(hs[-1][:, pre:].float().norm(dim=-1).cpu()); clsn.append(hs[-1][:, 0].float().norm(dim=-1).cpu())
            if nreg: regn.append(hs[-1][:, 1:pre].float().norm(dim=-1).cpu())
    last = torch.cat(last); pl = torch.cat(per_layer, 1)
    med = last.median().item()
    per_img_max_ratio = (last.max(1).values / last.median(1).values)
    rec = {"n_images": len(df), "patches_per_image": last.shape[1], "median_norm": med,
           "p99_norm": last.quantile(0.99).item(), "max_norm": last.max().item(),
           "frac_gt_150": (last > 150).float().mean().item(),
           "frac_gt_4x_median": (last > 4 * med).float().mean().item(),
           "imgs_with_gt_4x": (last > 4 * med).any(1).float().mean().item(),
           "per_img_max_over_median_median": per_img_max_ratio.median().item(),
           "cls_norm_median": torch.cat(clsn).median().item(),
           "layer_max_over_median": [(l.max() / l.median()).item() for l in pl],
           "hist_last": np.histogram(last.numpy().ravel(), bins=40)[0].tolist(),
           "hist_edges": np.histogram(last.numpy().ravel(), bins=40)[1].round(2).tolist()}
    if nreg: rec["register_norm_median"] = torch.cat(regn).median().item()
    out[name] = rec
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith("hist") and k != "layer_max_over_median"},
          [round(v, 1) for v in rec["layer_max_over_median"]], flush=True)
    del m, enc; torch.cuda.empty_cache()
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/v02/results/norms.json", "w"))
