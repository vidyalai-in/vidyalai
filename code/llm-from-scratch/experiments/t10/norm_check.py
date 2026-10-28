# Part 5's trap, measured once on real SFT micro-batches: the step's gradient with the correct
# normalization (divide by the step's total real targets) vs the naive mean of micro-batch means.
import sys, json, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import Conversations, plan, collate
from snapshots import load
data = Conversations("train", 20000)
groups = plan(data, 1, 4096, 1337)[0]
model = load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt').train()
res = []
for s in range(3):                                         # three steps of 4 micro-batches
    micro = [collate(data, g, T, "assistant") for g, T in groups[4 * s:4 * s + 4]]
    counts = [int((y != -1).sum()) for _, y in micro]
    def grad(weights):
        model.zero_grad(set_to_none=True)
        for (x, y), w in zip(micro, weights):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                (model(x.cuda(), y.cuda()) * w).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])
    g_ok = grad([n / sum(counts) for n in counts])
    g_naive = grad([1 / len(micro)] * len(micro))
    # reference: all four micro-batches' targets in one sum, divided once (FP32 sum of token losses)
    rel = ((g_naive - g_ok).norm() / g_ok.norm()).item()
    cosv = torch.nn.functional.cosine_similarity(g_naive, g_ok, dim=0).item()
    r = {"step": s, "rows": [x.shape[0] for x, _ in micro], "T": [x.shape[1] for x, _ in micro],
         "targets": counts, "rel_diff": rel, "cos": cosv,
         "naive_weight_per_token": [1 / len(micro) / n for n in counts]}
    print(json.dumps(r)); res.append(r)
json.dump(res, open('/home/amit/.cache/vidyalai-scratch/t10/post/norm_check.json', 'w'), indent=1)
