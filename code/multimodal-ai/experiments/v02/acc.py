# Exp 1b: ImageNet top-1 on the 24 Oxford-IIIT Pet breeds that have an exact ImageNet class,
# at input resolutions the models were and weren't trained on (ViT position embeddings interpolated bicubically).
# Exp 2: mean attention distance per layer and head (ViT-B/16, eager attention).
import sys, io, json, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vit as L
RES = [112, 160, 224, 288, 384, 448, 512, 640]
df = pd.read_parquet(L.pets_parquet("test"))
names = L.pets_class_names()
df = df[[names[i] in L.PETS_TO_IMAGENET for i in df["label"]]].reset_index(drop=True)
labels = np.array([L.PETS_TO_IMAGENET[names[i]] for i in df["label"]])
print("items", len(df), "classes", len(set(labels)), flush=True)
models = {"vit224": L.load_hf("google/vit-base-patch16-224"), "vit384": L.load_hf("google/vit-base-patch16-384"),
          "resnet50": L.load_hf("microsoft/resnet-50")}
NORM = {"vit224": "vit", "vit384": "vit", "resnet50": "imagenet", "vit224_nopos": "vit"}
# a copy of the 224 model with patch position embeddings replaced by their mean (positions removed)
nopos = L.load_hf("google/vit-base-patch16-224")
with torch.no_grad():
    pe = nopos.vit.embeddings.position_embeddings
    pe[:, 1:] = pe[:, 1:].mean(1, keepdim=True)
models["vit224_nopos"] = nopos
def prep(u8, r, norm):  # u8: (3,H,W) uint8 on GPU -> (3,r,r) normalised
    x = u8.float().unsqueeze(0) / 255
    h, w = x.shape[-2:]; s = r / min(h, w)
    x = F.interpolate(x, size=(max(r, round(h * s)), max(r, round(w * s))), mode="bicubic", antialias=True, align_corners=False)
    h, w = x.shape[-2:]; t, l = (h - r) // 2, (w - r) // 2
    x = x[..., t:t + r, l:l + r].clamp(0, 1)
    m, sd = L.MEAN_STD[norm]
    return ((x - torch.tensor(m, device=x.device).view(1, 3, 1, 1)) / torch.tensor(sd, device=x.device).view(1, 3, 1, 1))[0]
correct = {(m, r): [] for m in models for r in RES}; top5 = {(m, r): [] for m in models for r in RES}
CH = 64
sizes = []
with torch.no_grad():
    for s in range(0, len(df), CH):
        ims = [torch.from_numpy(np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))).permute(2, 0, 1).cuda()
               for b in df["image"][s:s + CH]]
        sizes += [tuple(i.shape[1:]) for i in ims]
        y = torch.tensor(labels[s:s + CH], device="cuda")
        for r in RES:
            for norm in ("vit", "imagenet"):
                x = torch.stack([prep(i, r, norm) for i in ims]).half()
                for mn, m in models.items():
                    if NORM[mn] != norm or (mn == "vit224_nopos" and r != 224):
                        continue
                    kw = {"interpolate_pos_encoding": True} if mn.startswith("vit") else {}
                    lg = m(pixel_values=x, **kw).logits.float()
                    correct[(mn, r)] += (lg.argmax(-1) == y).tolist()
                    top5[(mn, r)] += (lg.topk(5, -1).indices == y[:, None]).any(-1).tolist()
        print(s, flush=True)
out = {"n": len(df), "res": RES, "image_hw_median": np.median(np.array(sizes), 0).tolist(), "acc": {}, "flags": {}}
for (mn, r), v in correct.items():
    if not v: continue
    k, n = int(sum(v)), len(v); lo, hi = L.wilson(k, n)
    out["acc"][f"{mn}@{r}"] = {"top1": k / n, "lo": lo, "hi": hi, "k": k, "n": n, "top5": float(np.mean(top5[(mn, r)]))}
    out["flags"][f"{mn}@{r}"] = "".join("1" if c else "0" for c in v)
    print(mn, r, round(k / n, 4), round(lo, 4), round(hi, 4), "top5", round(float(np.mean(top5[(mn, r)])), 4), flush=True)
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/v02/results/acc.json", "w"))
# ---- Exp 2: mean attention distance ----
del models, nopos; torch.cuda.empty_cache()
eag = L.load_hf("google/vit-base-patch16-224", attn="eager")
md = {}
with torch.no_grad():
    for r in (224, 448):
        g = r // 16; acc = torch.zeros(12, 12, device="cuda"); cnt = 0
        for s in range(0, 256, 32):
            ims = [torch.from_numpy(np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))).permute(2, 0, 1).cuda()
                   for b in df["image"][s:s + 32]]
            x = torch.stack([prep(i, r, "vit") for i in ims]).half()
            o = eag(pixel_values=x, output_attentions=True, interpolate_pos_encoding=True)
            for li, a in enumerate(o.attentions):
                acc[li] += L.mean_attention_distance(a, g, g, 16) * len(ims)
            cnt += len(ims)
        md[r] = (acc / cnt).cpu().numpy().round(2).tolist()
        print(r, [round(float(np.mean(l)), 1) for l in md[r]], flush=True)
json.dump({"n_images": 256, "per_layer_head_px": md}, open("/home/amit/.cache/vidyalai-scratch/v02/results/attn_dist.json", "w"))
