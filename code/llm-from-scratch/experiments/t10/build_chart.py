# Writes diagrams/t10-sft-loss.py with the logged data embedded (from runs/sft88 and runs/sft88-all).
import json
def load(tag):
    L = [json.loads(l) for l in open(f'/home/amit/.cache/vidyalai-scratch/t10/runs/{tag}/log.jsonl')]
    tr = [round(r['loss'], 4) for r in L if 'loss' in r][:-1]          # drop the partial last step
    va = [(r['step'], round(r['val']['all'], 4)) for r in L if 'val' in r]
    return tr, va
A, B = load('sft88'), load('sft88-all')
head = '''# Generates public/diagrams/t10-sft-loss.svg and t10-sft-loss-dark.svg (data: t10-sft-from-scratch.md).
# Usage: python3 diagrams/t10-sft-loss.py public/diagrams/t10-sft-loss, then render the PNG with sharp at density 144.
# Data: fromscratch/sft.py logs (runs/sft88 = assistant-only loss, runs/sft88-all = loss on every token), 88M model
# from Part 7, 26,000 smol-smoltalk conversations, 290 full steps of about 59,000 padded tokens, peak lr 3e-4.
# TRAIN_*: training loss per optimizer step (mean over that run's trained tokens). VAL_*: held-out loss on the
# assistant tokens of 500 validation conversations, every 50 steps (step 0 = the base model).
'''
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/t10-sft-loss.py', 'w').write(head +
    f"TRAIN_A={A[0]}\nTRAIN_B={B[0]}\nVAL_A={A[1]}\nVAL_B={B[1]}\n" + open('/home/amit/.cache/vidyalai-scratch/t10/scripts/chart_body.py').read())
