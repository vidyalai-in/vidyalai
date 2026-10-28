# On-policy arm: sample K replies per prompt from the SFT model (temperature 1, no top-k), score them with the
# reward model, keep (best, worst) as (chosen, rejected), and save them as data/onpolicy/train_*.npy.
# Prompts: those of the first N training pairs, the same prompts the off-policy run starts with.
# usage: python onpolicy.py N K MAX_NEW
import sys, json, time, statistics
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
import torch
from helpers import SFT, uf_prompts, sample_all, END
from dpo import save_pairs
from sft import chat_tokenizer
from snapshots import load
N, K, MAX_NEW = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
tok = chat_tokenizer()
prompts = uf_prompts(N, 'train')
model = load(SFT)
# 64 rows per batch: 16 prompts x K=4. Time the first batch, then keep as many prompts as fit in GEN_BUDGET s.
from sampling import generate_batch
GEN_BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 750
per = 64 // K
samples, t1 = [], time.time()
for b in range(0, N, per):
    grp = [p for p in prompts[b:b + per] for _ in range(K)]
    mx = min(MAX_NEW, model.cfg.max_seq_len - max(len(p) for p in grp))
    samples += generate_batch(model, grp, mx, temperature=1.0, seed=100 + b)
    if b == 0:
        dt = time.time() - t1
        n_fit = min(N, int(GEN_BUDGET / dt) * per)
        print('first batch', round(dt, 2), 's; estimate for', N, 'prompts', round(dt * N / per), 's; keeping', n_fit, flush=True)
        N = n_fit
    if b + per >= N:
        break
prompts = prompts[:N]
rep = [p for p in prompts for _ in range(K)]
samples = samples[:len(rep)]
t_gen = time.time() - t0
del model
torch.cuda.empty_cache()
print('generated', len(samples), round(t_gen), flush=True)
from rm import RewardModel
rm = RewardModel()
texts = [(tok.decode(p[2:-2]), tok.decode(ids)) for p, (ids, _) in zip(rep, samples)]
scores = rm.score(texts)
t_rm = time.time() - t0 - t_gen
pairs, info = [], []
for i, p in enumerate(prompts):
    s = scores[i * K:(i + 1) * K]
    hi, lo = max(range(K), key=lambda k: s[k]), min(range(K), key=lambda k: s[k])
    c, r = samples[i * K + hi], samples[i * K + lo]
    seq = lambda x: p + x[0] + ([END] if x[1] else [])
    pairs.append((seq(c), seq(r), len(p)))
    info.append({"gap": s[hi] - s[lo], "chosen_ended": c[1], "rejected_ended": r[1],
                 "chosen_len": len(c[0]), "rejected_len": len(r[0])})
save_pairs('onpolicy', 'train', pairs)
stats = {"prompts": N, "k": K, "max_new": MAX_NEW, "gen_seconds": round(t_gen, 1), "rm_seconds": round(t_rm, 1),
         "ended_all": sum(e for _, e in samples) / len(samples),
         "chosen_ended": sum(x["chosen_ended"] for x in info) / N, "rejected_ended": sum(x["rejected_ended"] for x in info) / N,
         "chosen_len_mean": statistics.mean(x["chosen_len"] for x in info),
         "rejected_len_mean": statistics.mean(x["rejected_len"] for x in info),
         "chosen_longer": sum(x["chosen_len"] > x["rejected_len"] for x in info) / N,
         "gap_median": statistics.median(x["gap"] for x in info),
         "score_mean": statistics.mean(scores)}
json.dump(stats, open('/home/amit/.cache/vidyalai-scratch/t11/post/onpolicy_data.json', 'w'), indent=1)
json.dump([{"prompt": texts[i * K][0], "replies": [texts[i * K + k][1] for k in range(K)],
            "scores": scores[i * K:(i + 1) * K]} for i in range(min(N, 40))],
          open('/home/amit/.cache/vidyalai-scratch/t11/post/onpolicy_examples.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
