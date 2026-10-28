# Writes diagrams/t11-dpo-curves.py with the main run's logged data embedded (runs/dpo88/log.jsonl).
import json
L = [json.loads(l) for l in open('/home/amit/.cache/vidyalai-scratch/t11/runs/dpo88/log.jsonl')]
S = [r for r in L if 'loss' in r]
V = [r for r in L if 'val' in r]
beta = 0.1
tl = [round(r['loss'], 4) for r in S]
ta = [round(r['acc'], 4) for r in S]
vl = [(r['step'], round(r['val']['loss'], 4)) for r in V]
va = [(r['step'], round(r['val']['acc'], 4)) for r in V[1:]]          # step 0: all ties, undefined
dc = [(r['step'], round(r['val']['chosen_reward'] / beta, 3)) for r in V]
dr = [(r['step'], round(r['val']['rejected_reward'] / beta, 3)) for r in V]
head = '''# Generates public/diagrams/t11-dpo-curves.svg and t11-dpo-curves-dark.svg (data: t11-dpo-from-scratch.md).
# Usage: python3 diagrams/t11-dpo-curves.py public/diagrams/t11-dpo-curves, then render the PNG with sharp at density 144.
# Data: fromscratch/dpo.py log of runs/dpo88: DPO from Part 10's SFT weights (88M), 16,000 UltraFeedback pairs,
# 314 steps of about 51 pairs, beta 0.1, peak lr 1e-5. TRAIN_*: per optimizer step. VAL_*: the first 500 held-out
# pairs every 25 steps. VAL_DC / VAL_DR: mean change in log-probability (nats per reply) of the chosen / rejected
# replies against the reference, i.e. the implicit reward divided by beta. Step-0 accuracy is omitted (all ties).
'''
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/t11-dpo-curves.py', 'w').write(
    head + f"TRAIN_LOSS={tl}\nTRAIN_ACC={ta}\nVAL_LOSS={vl}\nVAL_ACC={va}\nVAL_DC={dc}\nVAL_DR={dr}\n" +
    open('/home/amit/.cache/vidyalai-scratch/t11/scripts/chart_curves_body.py').read())
