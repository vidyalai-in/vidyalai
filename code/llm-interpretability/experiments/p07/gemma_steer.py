"""Clamp one Gemma Scope latent (layer 12, 16k, L0 82) on Gemma 2 2B during generation and count concept words.
Clamping sets the latent's activation to c at every position after BOS and keeps the SAE's error term:
x <- x + (c - f_i(x)) * W_dec[i]. usage: gemma_steer.py <latent> <c1,c2,...> <keyword1,keyword2,...>
"""
import sys, json, re
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S

torch.set_grad_enabled(False)
lat = int(sys.argv[1]); cs = [float(c) for c in sys.argv[2].split(",")]; kws = sys.argv[3].lower().split(",")
PROMPTS = ["I went to the store and", "My favorite thing about weekends is", "The best way to start the day is",
           "Yesterday my friend told me that", "In the middle of the night, I", "The teacher asked the class to",
           "When I opened the door, I saw", "The most important lesson I learned was", "Here is a short story:",
           "The news this morning was about", "My grandmother always said that", "The city council voted to",
           "Scientists recently discovered that", "On our trip we decided to", "The new restaurant downtown",
           "After dinner we talked about", "The book I am reading is about", "She looked out the window and",
           "The meeting started with", "Tomorrow I plan to"]
g = S.GemmaRunner()
sae = S.load_gemma_scope(12, "16k", 82)
g.attach(12)
d = sae.W_dec[lat]


def clamp(c):
    def edit(hs):
        h = hs.float()
        sl = slice(1, None) if h.shape[1] > 1 else slice(None)
        x = h[:, sl]
        fi = sae.encode(x)[..., lat]
        h[:, sl] = x + (c - fi)[..., None] * d
        return h.to(hs.dtype)
    return edit


out = {"latent": lat, "keywords": kws, "runs": {}}
pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")", re.I)
for c in [None] + cs:
    g.edit = None if c is None else clamp(c)
    texts = []
    for p in PROMPTS:
        t = g.tok(p, return_tensors="pt").input_ids.cuda()
        o = g.model.generate(t, max_new_tokens=30, do_sample=False)
        texts.append(g.tok.decode(o[0, t.shape[1]:], skip_special_tokens=True))
    hits = sum(bool(pat.search(x)) for x in texts)
    out["runs"][str(c)] = {"hits": hits, "n": len(texts), "texts": texts}
    print("clamp", c, "hits", hits, "/", len(texts), "|", texts[0][:120].replace("\n", " "), flush=True)
g.edit = None
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/p07/runs/steer_{lat}.json", "w"), indent=1)
