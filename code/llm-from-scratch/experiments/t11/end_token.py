# What DPO did to the <|end|> token: log-probability of the closing <|end|> of every held-out chosen and
# rejected reply, under the SFT model and each DPO model; and the mean change on the other reply tokens.
# usage: python end_token.py tag=path ...
import sys, json
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
import torch, torch.nn.functional as F
from helpers import SFT
from dpo import Pairs, collate_pairs
from sft import plan, END
from snapshots import load
test = Pairs('ultrafeedback', 'test')
models = {"sft": SFT} | dict(a.split('=') for a in sys.argv[1:])
out = {}
for tag, path in models.items():
    model = load(path)
    s = {k: [0.0, 0] for k in ("end_chosen", "end_rejected", "body_chosen", "body_rejected")}
    with torch.no_grad():
        for group, T in plan(test, 1, 4096, seed=0)[0]:
            x, y = (t.cuda() for t in collate_pairs(test, group, T))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            lp = -F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-1, reduction='none').view_as(y)
            B = len(group)
            for half, name in ((slice(0, B), "chosen"), (slice(B, 2 * B), "rejected")):
                yy, ll = y[half], lp[half]
                e, b = yy == END, (yy != -1) & (yy != END)
                s["end_" + name][0] += ll[e].sum().item(); s["end_" + name][1] += int(e.sum())
                s["body_" + name][0] += ll[b].sum().item(); s["body_" + name][1] += int(b.sum())
    out[tag] = {k: v[0] / v[1] for k, v in s.items()}
    print(tag, out[tag], flush=True)
    del model
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t11/post/end_token.json', 'w'), indent=1)
