"""Run SD 1.5's safety checker on every saved frame and measure frame-to-frame CLIP similarity
(openai/clip-vit-base-patch32, MIT) for each clip directory under frames/."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import sd as SDL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
sc = StableDiffusionSafetyChecker.from_pretrained(SD, subfolder="safety_checker", torch_dtype=torch.float16).to("cuda")
fx = CLIPImageProcessor.from_pretrained(SD, subfolder="feature_extractor")
judge = SDL.ClipJudge()
out = {}
for d in sorted(Path("frames").iterdir()):
    if len(sys.argv) > 1 and d.name not in sys.argv[1:]: continue
    imgs = [Image.open(p).convert("RGB") for p in sorted(d.glob("*.png"))]
    inp = fx(imgs, return_tensors="pt").pixel_values.to("cuda", torch.float16)
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in imgs])
    with torch.no_grad():
        _, flags = sc(images=arr, clip_input=inp)
    e = judge.images(imgs)
    cons = (e[1:] * e[:-1]).sum(-1).numpy()
    first = (e[1:] * e[:1]).sum(-1).numpy()
    # mean absolute pixel change between consecutive frames, 0-255 scale (VBench-style flicker / motion)
    px = np.stack([np.asarray(im, dtype=np.float32) for im in imgs])
    mad = np.abs(px[1:] - px[:-1]).mean(axis=(1, 2, 3))
    out[d.name] = {"n": len(imgs), "flagged": int(sum(bool(f) for f in flags)), "flag_idx": [i for i, f in enumerate(flags) if f], "clip_consecutive": cons.tolist(),
                   "clip_to_first": first.tolist(), "mean_abs_diff": mad.tolist()}
    print(d.name, len(imgs), "flagged", out[d.name]["flagged"], "cos_next %.3f" % cons.mean(), "mad %.1f" % mad.mean(), flush=True)
json.dump(out, open("runs/frames_check%s.json" % ("_" + "_".join(sys.argv[1:]) if len(sys.argv) > 1 else ""), "w"), indent=1); print(out)
