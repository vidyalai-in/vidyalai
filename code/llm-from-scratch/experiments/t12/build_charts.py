# Writes diagrams/t12-pass-at-k.py and diagrams/t12-dynamics.py with data from t12/post and the main run's log.
import json
P = '/home/amit/.cache/vidyalai-scratch/t12/post/'
D = '/home/amit/AlpsCode/ai/vidyalai/diagrams/'
S = '/home/amit/.cache/vidyalai-scratch/t12/scripts/'
models = [('Part 10 SFT model', 'sft'), ('warm start', 'warm'), ('after GRPO', 'main'),
          ('mixed warm start', 'warm_mixed'), ('mixed + GRPO', 'grpo_mixed')]
ev = {t: json.load(open(P + f'eval_{t}.json')) for _, t in models}
def pk(t, split):
    d = ev[t][split]['pass_at_k']
    return {k: round(v, 4) for k, v in d.items() if int(k) <= 64}
panels = [(title, [pk(t, split) for _, t in models]) for title, split in
          (('1 to 3 digits (200 held-out)', 'test'), ('4 digits (100)', '4digit'), ('5 digits (100)', '5digit'))]
head = '''# Generates public/diagrams/t12-pass-at-k.svg and -dark.svg (data: t12-grpo-from-scratch.md).
# Usage: python3 diagrams/t12-pass-at-k.py public/diagrams/t12-pass-at-k, then render the PNG with sharp at density 144.
# Data: t12/scripts/evaluate.py. pass@k from n samples per problem at temperature 1 (n = 64 for the warm start and the
# GRPO model, 16 for the Part 10 SFT model, so its line stops at k = 16), unbiased estimator of Chen et al. (2021),
# strict checker. 'mixed' = warm start with 4,000 smol-smoltalk conversations added. Panels: held-out problems with 1 to 3 digits per number, and with exactly 4 and 5 digits.
'''
open(D + 't12-pass-at-k.py', 'w').write(head + 'MODELS=' + repr([(l, t) for l, t in models]) + '\nPANELS=' + repr(panels) + '\n' + open(S + 'chart_passk_body.py').read())
rec = [json.loads(l) for l in open('/home/amit/.cache/vidyalai-scratch/t12/runs/grpo_main/log.jsonl')]
tr = [r for r in rec if 'reward' in r]
r4 = lambda k: [round(r[k], 4) for r in tr]
dyn = [('Reply length, tokens (sampled replies)', [('', r4('len'))], 40, 60, (40, 45, 50, 55, 60), '{:g}'),
       ('Shares of replies and of groups', [('worked style', r4('worked')), ('zero-variance groups', r4('zero_var'))], 0, 1, (0, 0.25, 0.5, 0.75, 1), '{:.0%}'),
       ('KL from the reference, nats per reply token (k3 estimate)', [('', r4('kl_tok'))], 0, 0.06, (0, 0.02, 0.04, 0.06), '{:g}'),
       ('Entropy of the next-token distribution, nats per reply token', [('', r4('entropy'))], 0, 0.12, (0, 0.04, 0.08, 0.12), '{:g}')]
head = '''# Generates public/diagrams/t12-dynamics.svg and -dark.svg (data: t12-grpo-from-scratch.md).
# Usage: python3 diagrams/t12-dynamics.py public/diagrams/t12-dynamics, then render the PNG with sharp at density 144.
# Data: fromscratch/grpo.py log of t12/runs/grpo_main, one value per step over its 256 sampled replies.
'''
open(D + 't12-dynamics.py', 'w').write(head + 'PANELS=' + repr(dyn) + '\n' + open(S + 'chart_dyn_body.py').read())
print('ok')
