# Writes diagrams/t12-reward-curves.py with data from the GRPO run logs (main run and ablations).
# usage: python3 build_curves.py main=<run> label=<run> ...   (first pair is the main run)
import json, sys
R = '/home/amit/.cache/vidyalai-scratch/t12/runs/'
runs = []
for arg in sys.argv[1:]:
    label, run = arg.rsplit('=', 1)
    rec = [json.loads(l) for l in open(R + run + '/log.jsonl')]
    tr = [r for r in rec if 'reward' in r]
    te = [r for r in rec if 'test_acc' in r]
    runs.append((label, [round(r['reward'], 4) for r in tr], [(r['step'], round(r['test_acc'], 4)) for r in te]))
head = '''# Generates public/diagrams/t12-reward-curves.svg and -dark.svg (data: t12-grpo-from-scratch.md).
# Usage: python3 diagrams/t12-reward-curves.py public/diagrams/t12-reward-curves, then render the PNG with sharp at density 144.
# Data: fromscratch/grpo.py logs (t12/runs/<run>/log.jsonl). REWARD: mean reward of the step's 256 sampled
# replies (32 problems x 8) under each run's own checker (contains for the contains run). TEST: greedy accuracy on the 500 held-out problems, every 25 steps. All runs start from
# the warm-start weights (t12/runs/warm4000), strict checker for TEST. First run is the main one.
'''
body = open('/home/amit/.cache/vidyalai-scratch/t12/scripts/chart_curves_body.py').read()
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/t12-reward-curves.py', 'w').write(head + 'RUNS=' + repr(runs) + '\n' + body)
print('ok', [(l, len(a), len(b)) for l, a, b in runs])
