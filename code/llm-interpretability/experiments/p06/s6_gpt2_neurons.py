"""GPT-2 small MLP neurons on four unrelated kinds of text.
Domains (400 windows of 127 tokens each, plus <|endoftext|> at position 0, which is never scored):
  en  FineWeb-Edu held-out documents (ODC-By 1.0), from interp/data/p02
  de  random German Wikipedia articles (CC BY-SA 4.0), interp/data/p06/wiki-de.jsonl
  ko  random Korean Wikipedia articles and lead sections (CC BY-SA 4.0), interp/data/p06/wiki-ko*.jsonl
  py  Python source files from the CPython 3.12 standard library (PSF License), /usr/lib/python3.12
Activation: blocks.{l}.mlp.hook_post (after GELU), GPT-2 small fp32, TransformerLens without weight processing.
Pass 1: each neuron's max over all scored tokens, and how many neurons are > 0 per token.
Pass 2: tau = frac * max; per window, how many tokens have act > tau (saved), plus per neuron and domain
        the 5 strongest (value, token index) for examples."""
import glob, json, random, sys, time
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
DATA = "/home/amit/.cache/vidyalai-scratch/interp/data/"
NW, T = 400, 127
FRAC = 0.5
t0 = time.time()
model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cuda")
tok = model.tokenizer


def jsonl(path):
    return [json.loads(l)["text"] for l in open(path)]


rng = random.Random(0)
texts = {
    "en": jsonl(DATA + "p02/fineweb-edu-heldout600.jsonl"),
    "de": jsonl(DATA + "p06/wiki-de.jsonl"),
    "ko": jsonl(DATA + "p06/wiki-ko.jsonl") + jsonl(DATA + "p06/wiki-ko-leads.jsonl"),
}
pyfiles = sorted(glob.glob("/usr/lib/python3.12/*.py"))
rng.shuffle(pyfiles)
texts["py"] = [open(f, encoding="utf-8", errors="replace").read() for f in pyfiles[:80]]
DOMS = ["en", "de", "ko", "py"]

windows, meta = [], {}
for d in DOMS:
    ids = []
    for t in texts[d]:
        ids += tok.encode(t) + tok.encode("\n\n")
    chunks = [ids[i:i + T] for i in range(0, len(ids) - T, T)]
    pick = sorted(rng.sample(range(len(chunks)), NW))       # 400 windows spread over the stream
    windows += [[tok.bos_token_id] + chunks[i] for i in pick]
    meta[d] = dict(docs=len(texts[d]), tokens_available=len(ids), windows_available=len(chunks))
    print(d, meta[d], flush=True)
W = torch.tensor(windows, device="cuda")                    # [1600, 128], domain-major order
dom_of = torch.arange(len(DOMS), device="cuda").repeat_interleave(NW)
torch.save(dict(tokens=W.cpu(), doms=DOMS), RUN + "s6_windows.pt")

L, H, B = 12, 3072, 20
names = [f"blocks.{l}.mlp.hook_post" for l in range(L)]


def acts_for(batch):
    _, cache = model.run_with_cache(batch, names_filter=lambda n: n in names)
    return torch.stack([cache[n][:, 1:] for n in names])   # [L, B, T, H]


# pass 1
mx = torch.full((L, H), -1e9, device="cuda")
pos_count = torch.zeros(len(DOMS), L, device="cuda")       # sum over tokens of (#neurons > 0)
for i in range(0, len(W), B):
    a = acts_for(W[i:i + B])
    mx = torch.maximum(mx, a.amax((1, 2)))
    d = dom_of[i].item()
    pos_count[d] += (a > 0).sum((1, 2, 3)).float()
print("pass1", round(time.time() - t0), "s", flush=True)

tau = FRAC * mx
C = torch.zeros(len(W), L, H, dtype=torch.int16, device="cuda")
above_count = torch.zeros(len(DOMS), L, device="cuda")     # sum over tokens of (#neurons > tau)
K = 5
best_v = torch.full((len(DOMS), L, H, K), -1e9, device="cuda")
best_i = torch.zeros((len(DOMS), L, H, K), dtype=torch.long, device="cuda")
for i in range(0, len(W), B):
    a = acts_for(W[i:i + B])                                # [L, b, T, H]
    on = a > tau[:, None, None, :]
    C[i:i + B] = on.sum(2).permute(1, 0, 2).to(torch.int16)
    d = dom_of[i].item()
    above_count[d] += on.sum((1, 2, 3)).float()
    flat = a.reshape(L, -1, H)                              # token index = (i + b) * T + pos
    v, j = flat.topk(K, dim=1)                              # [L, K, H]
    v, j = v.permute(0, 2, 1), (j + i * T).permute(0, 2, 1)
    allv, alli = torch.cat([best_v[d], v], -1), torch.cat([best_i[d], j], -1)
    top = allv.topk(K, dim=-1)
    best_v[d], best_i[d] = top.values, torch.gather(alli, -1, top.indices)
print("pass2", round(time.time() - t0), "s", flush=True)

ntok = NW * T
torch.save(dict(C=C.cpu(), max=mx.cpu(), tau=tau.cpu(), frac=FRAC, best_v=best_v.cpu(), best_i=best_i.cpu(),
                doms=DOMS, NW=NW, T=T), RUN + "s6_counts.pt")
summary = dict(model="gpt2 (openai-community/gpt2), fp32, from_pretrained_no_processing", hook="mlp.hook_post",
               windows_per_domain=NW, tokens_per_window=T, meta=meta, frac=FRAC,
               mean_neurons_positive_per_token={d: (pos_count[k] / ntok).tolist() for k, d in enumerate(DOMS)},
               mean_neurons_above_tau_per_token={d: (above_count[k] / ntok).tolist() for k, d in enumerate(DOMS)},
               seconds=round(time.time() - t0, 1), peak_gpu_mb=torch.cuda.max_memory_allocated() // 2**20)
json.dump(summary, open(RUN + "s6_summary.json", "w"), indent=1)
print(json.dumps(summary)[:1500])
