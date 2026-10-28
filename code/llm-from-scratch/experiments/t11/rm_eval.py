# Reward-model checks. (1) How often the RM agrees with UltraFeedback's GPT-4-based labels on the 1,370
# held-out pairs. (2) RM scores of each model's replies to the 500 held-out prompts (saved by evaluate.py),
# and the per-prompt win rate of each model against the SFT model. A weak proxy: one 0.6B judge.
# usage: python rm_eval.py sft tag1 tag2 ...
import sys, json, time
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
from helpers import uf_prompts, mean_ci
from dpo import Pairs
from sft import chat_tokenizer
from rm import RewardModel
P = '/home/amit/.cache/vidyalai-scratch/t11/post/'
t0 = time.time()
tok, rm = chat_tokenizer(), RewardModel()
test = Pairs('ultrafeedback', 'test')
pt = lambda k: tok.decode(test.seq(k)[2:test.start[k] - 2].tolist())
rt = lambda k: tok.decode(test.seq(k)[test.start[k]:-1].tolist())
out = {}
if '--no-calib' not in sys.argv:
    sc = rm.score([(pt(2 * i), rt(2 * i)) for i in range(test.n)] + [(pt(2 * i + 1), rt(2 * i + 1)) for i in range(test.n)])
    out = {"rm_agree_uf": sum(sc[i] > sc[test.n + i] for i in range(test.n)) / test.n,
           "rm_chosen_mean": sum(sc[:test.n]) / test.n, "rm_rejected_mean": sum(sc[test.n:]) / test.n}
print(out, round(time.time() - t0), flush=True)
prompts = [tok.decode(p[2:-2]) for p in uf_prompts(500)]
tags = [t for t in sys.argv[1:] if not t.startswith('--')]
scores = {}
for tag in tags:
    g = json.load(open(P + f'{tag}_gens.json'))
    for kind in ("uf_sampled", "uf_greedy"):
        if kind in g:
            scores[(tag, kind)] = rm.score([(p, tok.decode(ids)) for p, (ids, e) in zip(prompts, g[kind])])
            m, ci = mean_ci(scores[(tag, kind)])
            out[f"{tag}/{kind}/mean"], out[f"{tag}/{kind}/ci"] = m, ci
for tag in tags[1:]:
    for kind in ("uf_sampled", "uf_greedy"):
        if (tag, kind) in scores and (tags[0], kind) in scores:
            a, b = scores[(tag, kind)], scores[(tags[0], kind)]
            wins = [1.0 if x > y else 0.5 if x == y else 0.0 for x, y in zip(a, b)]
            m, ci = mean_ci(wins)
            out[f"{tag}/{kind}/win_vs_{tags[0]}"], out[f"{tag}/{kind}/win_ci"] = m, ci
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(P + ('rm_eval_onpolicy.json' if '--no-calib' in sys.argv else 'rm_eval.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
