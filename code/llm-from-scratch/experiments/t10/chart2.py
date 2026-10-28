# Writes diagrams/t10-before-after.py with the measured numbers embedded.
import json
P='/home/amit/.cache/vidyalai-scratch/t10/post/'
b, a, c = (json.load(open(P+f'{t}.json')) for t in ('base', 'sft88', 'sft88-all'))
sg = json.load(open(P+'sampled_gen.json'))
def acc(d, t, k): return round(100 * d['bench'][t][k], 1), round(100 * d['bench'][t][k + '_ci'], 1)
rows = [("HellaSwag acc_norm", acc(b,'hellaswag','acc_norm'), acc(a,'hellaswag','acc_norm'), acc(c,'hellaswag','acc_norm'), 25.0),
        ("ARC-Easy acc", acc(b,'arc_easy','acc'), acc(a,'arc_easy','acc'), acc(c,'arc_easy','acc'), 25.0),
        ("PIQA acc", acc(b,'piqa','acc'), acc(a,'piqa','acc'), acc(c,'piqa','acc'), 50.0)]
head = '''# Generates public/diagrams/t10-before-after.svg and t10-before-after-dark.svg (data: t10-sft-from-scratch.md).
# Usage: python3 diagrams/t10-before-after.py public/diagrams/t10-before-after, then render the PNG with sharp at density 144.
# Data: Part 9's evals.py (zero-shot, lm-eval-harness formats) on the base 88M model and the two SFT models from
# fromscratch/sft.py: (name, (base, 95% CI), (assistant-only SFT, CI), (all-token SFT, CI), chance), in percent.
'''
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/t10-before-after.py','w').write(head + f"ROWS={rows}\n" + open('chart2_body.py').read())
