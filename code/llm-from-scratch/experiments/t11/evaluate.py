# Before/after evaluation of one model against the SFT reference.
# usage: python evaluate.py <weights.pt> <tag> [--bench] [--quick]
#  1. held-out UltraFeedback pairs (1,370): log-probs, implicit-reward accuracy vs the SFT reference
#  2. Part 10's fixed prompts through sft.chat (greedy, and temperature 0.8 / top-k 50 / seed 0), 200 tokens
#  3. reply statistics: 500 held-out UF prompts (greedy and sampled, up to 512 tokens) and Part 10's
#     100 smol-smoltalk prompts (greedy and sampled, up to 256 tokens)
#  4. KL(policy || reference) on 300 held-out UF prompts: samples at temperature 1, summed log-ratio
#  5. Part 9's benchmarks (--bench)
import sys, json, time
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
import numpy as np, torch
from helpers import SFT, uf_prompts, smol_prompts, sample_all, reply_logps, reply_stats, mean_ci
from dpo import Pairs, all_logps
from sft import chat, chat_tokenizer
from snapshots import load
path, tag = sys.argv[1], sys.argv[2]
bench, quick, lite = '--bench' in sys.argv, '--quick' in sys.argv, '--lite' in sys.argv
P = '/home/amit/.cache/vidyalai-scratch/t11/post/'
t0 = time.time()
model, tok = load(path), chat_tokenizer()
out = {"path": path, "tag": tag}

# 1. held-out pairs
test = Pairs('ultrafeedback', 'test')
ref = np.load(f'{test.prefix}_ref.npy')
lp, ntok = (t.numpy() for t in all_logps(model, test))
d = lp - ref
out["pairs"] = {"n": test.n, "chosen_per_tok": float(lp[:, 0].sum() / ntok[:, 0].sum()),
                "rejected_per_tok": float(lp[:, 1].sum() / ntok[:, 1].sum()),
                "chosen_dlogp_mean": float(d[:, 0].mean()), "rejected_dlogp_mean": float(d[:, 1].mean()),
                "implicit_acc": float((d[:, 0] > d[:, 1]).mean()),
                "logratio_margin_mean": float((d[:, 0] - d[:, 1]).mean()),
                "acc_sum": float((lp[:, 0] > lp[:, 1]).mean()),
                "acc_per_tok": float((lp[:, 0] / ntok[:, 0] > lp[:, 1] / ntok[:, 1]).mean()),
                "frac_chosen_down": float((d[:, 0] < 0).mean()), "frac_rejected_down": float((d[:, 1] < 0).mean())}
np.save(P + f'{tag}_test_logps.npy', lp)
print(tag, 'pairs', out["pairs"], round(time.time() - t0), flush=True)

# 2. fixed prompts, exactly as Part 10
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
if not (quick or lite):
    out["fixed"] = {}
    for k, msgs in PROMPTS.items():
        g = chat(model, tok, msgs, max_new=200)
        s = chat(model, tok, msgs, max_new=200, temperature=0.8, top_k=50, seed=0)
        out["fixed"][k] = {"greedy": g, "sampled": s}
        print(k, '|G|', g[0][:200].replace('\n', ' '), g[1:], '\n   |S|', s[0][:200].replace('\n', ' '), s[1:], flush=True)

# 3. reply statistics (skipped with --lite: held-out pairs and KL only)
up = uf_prompts(500)
gens = {}
if not lite:
    gens["uf_sampled"] = sample_all(model, up, 512, temperature=0.8, top_k=50, seed=0)
if not (quick or lite):
    gens["uf_greedy"] = sample_all(model, up, 512, temperature=0)
    sp = smol_prompts(100)
    gens["smol_greedy"] = sample_all(model, sp, 256, temperature=0)
    gens["smol_sampled"] = sample_all(model, sp, 256, temperature=0.8, top_k=50, seed=0)
out["gen"] = {k: reply_stats(v) for k, v in gens.items()}
if gens:
    json.dump({k: [[ids, e] for ids, e in v] for k, v in gens.items()}, open(P + f'{tag}_gens.json', 'w'))
print(tag, 'gen', out["gen"], round(time.time() - t0), flush=True)

# 4. KL from the reference on held-out prompts, from the policy's own samples at temperature 1
kp = uf_prompts(300)
samples = sample_all(model, kp, 512, temperature=1.0, seed=1)
pol = reply_logps(model, kp, samples)
del_ref = load(SFT)
rf = reply_logps(del_ref, kp, samples)
del del_ref
kl = [a - b for (a, _), (b, _) in zip(pol, rf)]
m, ci = mean_ci(kl)
out["kl"] = {"n": len(kl), "per_reply": m, "per_reply_ci": ci,
             "per_token": sum(kl) / sum(n for _, n in pol), "t1_stats": reply_stats(samples),
             "policy_per_tok": sum(a for a, _ in pol) / sum(n for _, n in pol)}
print(tag, 'kl', out["kl"], round(time.time() - t0), flush=True)

# 5. benchmarks
if bench:
    from evals import OurLM, load_task, score_task, summarize
    del model
    lm = OurLM(path)
    out["bench"] = {}
    for task in ("hellaswag", "arc_easy", "piqa"):
        out["bench"][task] = summarize(score_task(lm, load_task(task)))
        print(task, out["bench"][task], round(time.time() - t0), flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(P + f'{tag}.json', 'w'), indent=1)
print('done', out["seconds"])
