# Collect residual-stream activations for the POS and truth probing sets, plus the model's own
# log-probability of each statement's country. Usage: collect.py gpt2|qwen
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.probes import load_model, tokenize_with_spans, collect_activations, bos_id

R = "/home/amit/.cache/vidyalai-scratch/p03/runs"
which = sys.argv[1]
name, dtype = {"gpt2": ("gpt2", torch.float32), "qwen": ("Qwen/Qwen2.5-0.5B", torch.bfloat16)}[which]
t0 = time.time()
model = load_model(name, dtype=dtype)
tok = model.tokenizer
print("loaded", name, dtype, f"{time.time()-t0:.1f}s", flush=True)

THE = {"United States", "United Kingdom", "United Arab Emirates", "Philippines",
       "Democratic Republic of the Congo", "Dominican Republic"}
def cname(c): return ("the " + c) if c in THE else c
TEMPLATES = {"cities": "The city of {city} is in {country}.", "alt": "{city} is a city in {country}."}

# ---- POS
pos = json.load(open(f"{R}/pos_data.json"))
for split in ["train", "dev", "test"]:
    toks, poss, dropped = [], [], 0
    for s in pos[split]:
        ids, p = tokenize_with_spans(tok, s["text"], [(a, b) for _, _, a, b in s["words"]])
        if min(p) < 0:
            dropped += 1
        toks.append(ids); poss.append([max(x, 1) for x in p])
    info = collect_activations(model, toks, poss, f"{R}/acts/{which}/pos_{split}", batch_size=32,
                               meta={"model": name, "dtype": str(dtype), "dropped_sentences_with_unaligned_words": dropped,
                                     "n_tokens": sum(len(t) for t in toks)})
    print("pos", split, info, f"{time.time()-t0:.1f}s", flush=True)

# ---- truth
items = json.load(open(f"{R}/truth_data.json"))
for tname, tmpl in TEMPLATES.items():
    toks, poss = [], []
    for it in items:
        text = tmpl.format(city=it["city"], country=cname(it["country"]))
        ids = [bos_id(tok)] + tok(text, add_special_tokens=False)["input_ids"]
        toks.append(ids); poss.append([len(ids) - 1])  # the final "." token
    info = collect_activations(model, toks, poss, f"{R}/acts/{which}/truth_{tname}", batch_size=64,
                               meta={"model": name, "template": tmpl})
    print("truth", tname, info, f"{time.time()-t0:.1f}s", flush=True)

# ---- the model's own preference: log P(country tokens + ".") after "The city of X is in"
lp = []
with torch.no_grad():
    for it in items:
        pre = [bos_id(tok)] + tok(f"The city of {it['city']} is in", add_special_tokens=False)["input_ids"]
        full = [bos_id(tok)] + tok(f"The city of {it['city']} is in {cname(it['country'])}.", add_special_tokens=False)["input_ids"]
        assert full[:len(pre)] == pre
        logits = model(torch.tensor([full], device="cuda")).float().log_softmax(-1)[0]
        tgt = torch.tensor(full[len(pre):], device="cuda")
        lp.append(logits[len(pre) - 1:len(full) - 1].gather(1, tgt[:, None]).sum().item())
json.dump(lp, open(f"{R}/acts/{which}/truth_logprob.json", "w"))
print("done", f"{time.time()-t0:.1f}s", "peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
