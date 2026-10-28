# How far are UltraFeedback's replies from what the SFT model writes? Per-token log-probability under the
# SFT model of (a) the chosen and rejected replies of the 1,370 held-out pairs, (b) the SFT model's own
# samples (temperature 1, no top-k) for the same prompts. Also: how often the SFT model already ranks
# chosen above rejected, by total and by per-token log-probability.
import sys, json, time
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
import numpy as np, torch
from helpers import SFT, uf_prompts, sample_all, reply_logps, reply_stats
from dpo import Pairs, all_logps
from snapshots import load
t0 = time.time()
model = load(SFT)
test = Pairs('ultrafeedback', 'test')
lp, ntok = all_logps(model, test)
lp, ntok = lp.numpy(), ntok.numpy()
ref = np.load(f'{test.prefix}_ref.npy')
out = {"n_pairs": test.n, "ref_check_maxdiff": float(np.abs(ref - lp).max()),
       "chosen_per_tok": float(lp[:, 0].sum() / ntok[:, 0].sum()),
       "rejected_per_tok": float(lp[:, 1].sum() / ntok[:, 1].sum()),
       "chosen_len_mean": float(ntok[:, 0].mean()), "rejected_len_mean": float(ntok[:, 1].mean()),
       "acc_sum": float((lp[:, 0] > lp[:, 1]).mean()),
       "acc_per_tok": float((lp[:, 0] / ntok[:, 0] > lp[:, 1] / ntok[:, 1]).mean()),
       "chosen_longer": float((ntok[:, 0] > ntok[:, 1]).mean())}
print(out, round(time.time() - t0), flush=True)
# per-token distribution of the chosen replies: share of tokens the SFT model finds very unlikely
from dpo import collate_pairs
from sft import plan
import torch.nn.functional as F
tok_lps = {"chosen": [], "rejected": []}
with torch.no_grad():
    for group, T in plan(test, 1, 4096, seed=0)[0]:
        x, y = collate_pairs(test, group, T)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x.cuda())
        l = -F.cross_entropy(logits.float().flatten(0, 1), y.cuda().flatten(), ignore_index=-1, reduction='none').view_as(y)
        m = y != -1
        B = len(group)
        tok_lps["chosen"].append(l[:B][m[:B].cuda()].cpu()); tok_lps["rejected"].append(l[B:][m[B:].cuda()].cpu())
for k in tok_lps:
    v = torch.cat(tok_lps[k])
    out[f"{k}_tok_below_-10"] = (v < -10).float().mean().item()
    out[f"{k}_tok_median"] = v.median().item()
# own samples at temperature 1
prompts = uf_prompts(test.n)
samples = sample_all(model, prompts, 512, temperature=1.0, seed=0)
sl = reply_logps(model, prompts, samples)
out["own_per_tok"] = sum(s for s, _ in sl) / sum(n for _, n in sl)
out["own_stats"] = reply_stats(samples)
own = []
with torch.no_grad():
    pass
print(out, round(time.time() - t0), flush=True)
# token-level distribution for own samples
from helpers import END
vals = []
for p, (ids, e) in zip(prompts[:400], samples[:400]):
    s = p + ids + ([END] if e else [])
    x = torch.tensor([s[:-1]], device='cuda'); y = torch.tensor([s[1:]], device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        l = model(x)[0].float().log_softmax(-1).gather(-1, y[0][:, None])[:, 0]
    vals.append(l[len(p) - 1:].cpu())
v = torch.cat(vals)
out["own_tok_below_-10"] = (v < -10).float().mean().item(); out["own_tok_median"] = v.median().item()
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t11/post/offpolicy.json', 'w'), indent=1)
json.dump([[p, s] for p, s in zip(prompts[:50], samples[:50])], open('/home/amit/.cache/vidyalai-scratch/t11/post/offpolicy_samples.json', 'w'))
print(json.dumps(out, indent=1))
