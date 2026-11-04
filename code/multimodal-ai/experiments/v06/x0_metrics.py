# How close the predicted clean latent at each DDIM step is to the final latent (CPU only).
import json, numpy as np
from PIL import Image
out = {}
for tag in ["lighthouse", "red"]:
    fin = np.load(f"runs/x0/{tag}_lat_19.npy")  # latent entering the last step; compare with x0 at last step instead
    x0f = np.load(f"runs/x0/{tag}_x0lat_19.npy")
    fimg = np.asarray(Image.open(f"runs/x0/{tag}_final.png"), np.float32)
    rows = []
    for i in range(20):
        x0 = np.load(f"runs/x0/{tag}_x0lat_{i:02d}.npy")
        corr = float(np.corrcoef(x0.ravel(), x0f.ravel())[0, 1])
        im = np.asarray(Image.open(f"runs/x0/{tag}_x0_{i:02d}.png"), np.float32)
        psnr = float(10 * np.log10(255**2 / ((im - fimg) ** 2).mean()))
        rows.append(dict(step=i + 1, latent_corr=corr, psnr_to_final=psnr))
    out[tag] = rows
json.dump(out, open("runs/x0_metrics.json", "w"), indent=1)
for tag in out:
    print(tag, [(r["step"], round(r["latent_corr"], 3), round(r["psnr_to_final"], 1)) for r in out[tag]])
