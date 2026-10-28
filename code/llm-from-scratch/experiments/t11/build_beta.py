# Writes diagrams/t11-beta.py with the beta-sweep measurements embedded (t11/post/<run>.json from evaluate.py).
import json
P = '/home/amit/.cache/vidyalai-scratch/t11/post/'
rows = []
for tag, name, sub in (('beta_0.05', 'β = 0.05', '2,000 pairs, 40 steps'), ('pilot_1e-5', 'β = 0.1', '2,000 pairs, 40 steps'),
                       ('beta_0.5', 'β = 0.5', '2,000 pairs, 40 steps'), ('dpo88', 'β = 0.1, main run', '16,000 pairs, 314 steps')):
    d = json.load(open(P + tag + '.json'))
    rows.append((name, sub, round(d['kl']['per_reply'], 2), round(d['kl']['per_reply_ci'], 2),
                 round(d['pairs']['chosen_dlogp_mean'], 2), round(d['pairs']['rejected_dlogp_mean'], 2),
                 round(d['kl']['t1_stats']['len_mean'], 1)))
sft = round(json.load(open(P + 'sft.json'))['kl']['t1_stats']['len_mean'], 1)
head = '''# Generates public/diagrams/t11-beta.svg and t11-beta-dark.svg (data: t11-dpo-from-scratch.md).
# Usage: python3 diagrams/t11-beta.py public/diagrams/t11-beta, then render the PNG with sharp at density 144.
# Data: DPO runs from Part 10's SFT model at peak lr 1e-5 (fromscratch/dpo.py), measured by t11/scripts/evaluate.py.
# ROWS: (label, run, KL(policy || SFT) per reply in nats from 300 held-out prompts sampled at temperature 1, its 95% CI,
# mean change in log-probability of the 1,370 held-out chosen and rejected replies in nats, mean length in tokens of
# the temperature-1 replies, capped at 512). SFT_LEN: the SFT model's mean length on the same prompts.
'''
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/t11-beta.py', 'w').write(
    head + f"ROWS={rows}\nSFT_LEN={sft}\n" + open('/home/amit/.cache/vidyalai-scratch/t11/scripts/chart_beta_body.py').read())
