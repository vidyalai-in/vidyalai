# Before/after evaluation of one model: held-out response loss, fixed prompts (greedy and sampled),
# reply statistics on held-out prompts, special-token embeddings, and Part 9's benchmarks.
# usage: python evaluate.py <weights.pt> <tag> [--bench] [--gen N]
import sys, json, time, statistics, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import Conversations, response_loss, chat, chat_tokenizer, END
from snapshots import load
path, tag = sys.argv[1], sys.argv[2]
bench, n_gen = "--bench" in sys.argv, int(sys.argv[sys.argv.index("--gen") + 1]) if "--gen" in sys.argv else 0
out = {"path": path, "tag": tag}
t0 = time.time()
model, tok = load(path), chat_tokenizer()
val = Conversations("val")
out["val"] = response_loss(model, val); print(tag, "val", out["val"], round(time.time() - t0), flush=True)

# special-token embedding rows
E = model.tok_emb.weight.detach().float()
sp = E[:8]; nrm = E.norm(dim=1)
cos = torch.nn.functional.normalize(sp, dim=1) @ torch.nn.functional.normalize(sp, dim=1).T
out["emb"] = {"special_norms": nrm[:8].tolist(), "median_norm": nrm.median().item(),
              "special_pairwise_cos_mean": ((cos.sum() - 8) / 56).item(),
              "chat_pairwise_cos": cos[1:5, 1:5].tolist()}

U = lambda s: {"role": "user", "content": s}
PROMPTS = {
  "capital": [U("What is the capital of France?")],
  "sky": [U("Why is the sky blue?")],
  "sleep": [U("Give me three tips for sleeping better.")],
  "poem": [U("Write a short poem about the ocean.")],
  "formal": [U("Rewrite this sentence to sound more formal: hey, can u send me the report by tmrw?")],
  "photosynthesis": [U("Explain photosynthesis to a ten-year-old in two sentences.")],
  "unknowable": [U("What was the weather like in Pune yesterday?")],
  "name": [U("What is my name?")],
  "math": [U("What is 17 + 25?")],
}
out["fixed"] = {}
for k, msgs in PROMPTS.items():
    g = chat(model, tok, msgs, max_new=200)
    s = chat(model, tok, msgs, max_new=200, temperature=0.8, top_k=50, seed=0)
    out["fixed"][k] = {"greedy": g, "sampled": s}
    print(k, "|G|", g[0][:300].replace("\n", " "), g[1:], "\n   |S|", s[0][:300].replace("\n", " "), s[1:], flush=True)
# multi-turn: the model's own first reply stays in the context
conv = [U("I'm planning a trip to Japan in spring. What should I see?")]
r1 = chat(model, tok, conv, max_new=200)
conv += [{"role": "assistant", "content": r1[0]}, U("Which of those would be best for young kids?")]
r2 = chat(model, tok, conv, max_new=200)
out["multiturn"] = [r1, r2]; print("multiturn", r1, r2, flush=True)

# reply statistics on held-out prompts: the first user turn (and system message) of val conversations
if n_gen:
    import numpy as np
    stats = []
    for i in range(n_gen):
        ids = val.ids[val.off[i]:val.off[i + 1]].tolist()
        cut = ids.index(3)                               # up to the first <|assistant|>
        idx = torch.tensor([ids[:cut + 1]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.generate(idx, 256, temperature=0.0, eos_id=END)[0, cut + 1:].tolist()
        ended = o[-1] == END
        body = o[:-1] if ended else o
        grams = [tuple(body[j:j + 4]) for j in range(len(body) - 3)]
        stats.append({"ended": ended, "len": len(body), "rep4": 1 - len(set(grams)) / max(1, len(grams))})
    out["gen"] = {"n": n_gen, "ended": sum(s["ended"] for s in stats) / n_gen,
                  "len_mean": statistics.mean(s["len"] for s in stats),
                  "len_median": statistics.median(s["len"] for s in stats),
                  "len_ended_median": statistics.median([s["len"] for s in stats if s["ended"]] or [0]),
                  "rep4_mean": statistics.mean(s["rep4"] for s in stats)}
    print("gen", out["gen"], round(time.time() - t0), flush=True)

if bench:
    from evals import OurLM, load_task, score_task, summarize
    lm = OurLM(path)                                   # Part 9's wrapper, unchanged
    out["bench"] = {}
    for task in ("hellaswag", "arc_easy", "piqa"):
        out["bench"][task] = summarize(score_task(lm, load_task(task)))
        print(task, out["bench"][task], round(time.time() - t0), flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/t10/post/{tag}.json", "w"), indent=1)
print("done", out["seconds"])
