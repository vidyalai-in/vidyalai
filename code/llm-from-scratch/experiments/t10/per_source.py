# Held-out response loss by data source, for the base model and both SFT models (full 2,000 validation conversations).
import sys, json, torch, collections
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import Conversations, collate, END
from snapshots import load
val = Conversations("val")
src = json.load(open('/home/amit/.cache/vidyalai-scratch/fromscratch/data/smol-smoltalk/val_sources.json'))
paths = {"base": "/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt",
         "sft88": "/home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt",
         "sft88-all": "/home/amit/.cache/vidyalai-scratch/t10/runs/sft88-all/model_final.pt"}
out = {}
for tag, p in paths.items():
    model = load(p)
    sums = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])   # content nats, n, prompt nats, n
    with torch.no_grad():
        for i in range(val.n):
            x, y = collate(val, [i], int(val.lengths[i]), "assistant")
            _, ya = collate(val, [i], int(val.lengths[i]), "all")
            x, y, ya = x.cuda(), y.cuda(), ya.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x).float()
            nll = torch.nn.functional.cross_entropy(lg[0], ya[0], reduction="none")
            resp = (y[0] != -1) & (y[0] != END)
            prm = (y[0] == -1) & (ya[0] > 7)                    # prompt text tokens (not special tokens)
            for s in (src[i], "all"):
                sums[s][0] += nll[resp].sum().item(); sums[s][1] += int(resp.sum())
                sums[s][2] += nll[prm].sum().item(); sums[s][3] += int(prm.sum())
    out[tag] = {s: {"response": v[0] / v[1], "prompt": v[2] / max(1, v[3]), "n_resp": v[1], "n_prompt": v[3]} for s, v in sums.items()}
    print(tag, json.dumps({s: (round(v["response"], 3), round(v["prompt"], 3)) for s, v in out[tag].items()}), flush=True)
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t10/post/per_source.json', 'w'), indent=1)
