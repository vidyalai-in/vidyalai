"""Topic steering on base Gemma 2 2B: Gemma Scope latent 7498 ("Egypt", Part 7) vs a difference-of-means
direction for the same concept vs a random direction, added at the residual stream after layer 12
(= input of layer 13, the Gemma Scope layer_12 SAE's site), with a dose-response sweep.
usage: egypt.py"""
import sys, json, re, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.steering import Steerer, repetition, wilson, paired_boot
from lab import sae as SA
torch.set_grad_enabled(False)
R = "/home/amit/.cache/vidyalai-scratch/p09"
P7 = ["I went to the store and", "My favorite thing about weekends is", "The best way to start the day is",
      "Yesterday my friend told me that", "In the middle of the night, I", "The teacher asked the class to",
      "When I opened the door, I saw", "The most important lesson I learned was", "Here is a short story:",
      "The news this morning was about", "My grandmother always said that", "The city council voted to",
      "Scientists recently discovered that", "On our trip we decided to", "The new restaurant downtown",
      "After dinner we talked about", "The book I am reading is about", "She looked out the window and",
      "The meeting started with", "Tomorrow I plan to"]
NEW = ["The best part of my job is", "Last weekend my family", "The museum's newest exhibit", "My neighbor just bought",
       "The documentary we watched", "For my birthday I want", "The old photograph showed", "Our science project is about",
       "The weather forecast says", "The first thing I noticed was", "He opened the letter and read", "The children played",
       "At the bottom of the box we found", "The most interesting class this year", "My brother's favorite movie",
       "The podcast episode was about", "During the summer holidays", "The professor began the lecture by",
       "If I could travel anywhere, I", "The painting on the wall", "The festival this year", "Our garden is full of",
       "The website explained that", "The tour guide pointed at", "She wrote in her diary that", "The magazine article argued that",
       "We spent the afternoon", "The village was known for", "On the radio they were discussing", "The recipe calls for"]
PROMPTS = P7 + NEW
KW = re.compile(r"\b(egypt|nile|pharaoh|cairo|pyramid)", re.I)
TEMPL = ["Last summer we traveled to {X}.", "My history class is studying {X} this week.", "She wrote a long report about {X}.",
         "The documentary was about the culture of {X}.", "He has always wanted to visit {X}.", "The museum opened a new gallery on {X}.",
         "Our book club is reading a novel set in {X}.", "The food in {X} was the highlight of the trip.",
         "They showed us old photographs of {X}.", "The professor is an expert on the history of {X}.",
         "I just finished a travel guide to {X}.", "The news report focused on {X}.", "We learned about the rivers of {X}.",
         "My grandfather told stories about {X}.", "The students made a poster about {X}.", "Tourists from all over the world visit {X}.",
         "The podcast episode explored the ancient past of {X}.", "Her favorite topic at school was {X}.",
         "The exhibition included maps of {X}.", "There is a famous monument in {X}."]
CTRL = ["France", "Japan", "Brazil", "Canada", "India", "Mexico", "Kenya", "Norway"]
t0 = time.time()
s = Steerer("/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16")
sae = SA.load_gemma_scope(12, "16k", 82)
lat = 7498
d_sae = sae.W_dec[lat].float().cpu()
print("SAE decoder row norm", d_sae.norm().item(), flush=True)

def mean_after(texts, word_idx_fn):
    """mean residual (input of layer 13) over tokens from the place name to the end, per sentence."""
    vs = []
    for t in texts:
        enc = s.tok(t, return_tensors="pt").to("cuda")           # adds <bos>
        start = word_idx_fn(t)
        A = s.collect([t], layers=[13], n_last=enc.input_ids.shape[1], add_special_tokens=True)[0, :, 0]  # [T, d]
        vs.append(A[start:].mean(0))
    return torch.stack(vs).mean(0)

def start_of(t, X):
    pre = t[:t.index(X)].rstrip()
    return len(s.tok(pre).input_ids)                               # tokens before the name (incl. <bos>)

eg = mean_after([x.format(X="Egypt") for x in TEMPL], lambda t: start_of(t, "Egypt"))
ct = torch.stack([mean_after([x.format(X=c) for x in TEMPL], lambda t, c=c: start_of(t, c)) for c in CTRL]).mean(0)
d_md = eg - ct
rand = torch.randn(d_md.shape, generator=torch.Generator().manual_seed(0))
cos = torch.nn.functional.cosine_similarity(d_md, d_sae, dim=0).item()
print("cos(meandiff, SAE latent)", cos, "meandiff norm", d_md.norm().item(), flush=True)
# random-direction cosine baseline
rc = [abs(torch.nn.functional.cosine_similarity(torch.randn(d_md.shape), d_sae, dim=0).item()) for _ in range(1000)]

docs = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")["eval"].long()[:100]
# typical residual norm at this site on real text (positions after <bos>)
A = s.collect([s.tok.decode(docs[i, 1:].tolist()) for i in range(20)], layers=[13], n_last=200, add_special_tokens=True)
res_norm = A.norm(dim=-1).mean().item()
print("mean residual norm (input of layer 13, FineWeb-Edu)", res_norm, flush=True)
# SAE activation of latent on the mean-diff-steered vs. projections: how large is latent 7498 on Egypt text?
units = {"sae": d_sae / d_sae.norm(), "meandiff": d_md / d_md.norm(), "random": rand / rand.norm()}
strengths = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
base_ce = s.loss(docs)
rows, texts = [], {}
for name, u in units.items():
    for k in strengths:
        if k == 0 and name != "sae":
            continue
        v = u * k * res_norm
        with s.edits(None if k == 0 else (lambda m: m.add(13, v))):
            g = s.generate(PROMPTS, max_new_tokens=40, batch=50, add_special_tokens=True)
            ce = s.loss(docs) if k > 0 else base_ce
        hits = sum(bool(KW.search(x)) for x in g)
        row = dict(direction=name if k else "none", strength=k, hits=hits, n=len(g), pct_ci=wilson(hits, len(g)),
                   ce=float(ce.mean()), d_ce=paired_boot(base_ce, ce), repetition=float(np.mean([repetition(x) for x in g])))
        rows.append(row); texts[f"{name}/{k}"] = g
        print(name, k, "hits", hits, "/", len(g), "dCE", round(row["d_ce"][0], 3), "rep", round(row["repetition"], 3),
              "|", g[12][:100].replace("\n", " "), f"{time.time()-t0:.0f}s", flush=True)
json.dump({"latent": lat, "layer_site": "input of layer 13 = Gemma Scope layer_12 resid_post", "sae_row_norm": d_sae.norm().item(),
           "cos_meandiff_sae": cos, "random_cos_abs_mean": float(np.mean(rc)), "random_cos_abs_max": float(np.max(rc)),
           "meandiff_norm": d_md.norm().item(), "res_norm": res_norm, "n_prompts": len(PROMPTS), "n_docs": len(docs),
           "templates": len(TEMPL), "controls": CTRL, "rows": rows, "seconds": time.time() - t0},
          open(f"{R}/post/egypt_sweep.json", "w"), indent=1)
json.dump(texts, open(f"{R}/runs/egypt_texts.json", "w"), indent=1)
