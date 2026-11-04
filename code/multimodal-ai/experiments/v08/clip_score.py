"""CLIP score (100 x cosine, openai/clip-vit-base-patch32) of the saved aMUSEd images against their prompts, on CPU."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/v08")
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from prompts import PROMPTS
torch.set_num_threads(8)
D = Path("/home/amit/.cache/vidyalai-scratch/v08/amused")
def _feat(o):
    return o if isinstance(o, torch.Tensor) else o.pooler_output
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
files = sorted(D.glob("*.png"))
meta = []
with torch.no_grad():
    te = _feat(clip.get_text_features(**proc(text=PROMPTS, return_tensors="pt", padding=True)))
    te = te / te.norm(dim=-1, keepdim=True)
    for i in range(0, len(files), 50):
        ch = files[i:i + 50]
        ie = _feat(clip.get_image_features(pixel_values=proc(images=[Image.open(f).convert("RGB") for f in ch], return_tensors="pt")["pixel_values"]))
        ie = ie / ie.norm(dim=-1, keepdim=True)
        sims = ie @ te.t()
        for f, row in zip(ch, sims):
            s, g, seed, p = map(int, re.match(r"s(\d+)_g(\d+)_seed(\d+)_p(\d+)", f.stem).groups())
            others = torch.cat([row[:p], row[p + 1:]])
            meta.append({"steps": s, "guidance": g, "seed": seed, "prompt_id": p, "file": f.name, "clip": 100 * row[p].item(),
                         "clip_mismatch": 100 * others.mean().item(), "clip_rank": int((row > row[p]).sum()) + 1})
with open(D.parent / "runs" / "amused_clip.jsonl", "w") as f:
    for m in meta:
        f.write(json.dumps(m) + "\n")
print(len(meta))
