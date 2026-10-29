# Part 9's measurements for the S15 students: loss on the whole validation shard, bits per byte on
# 2M tokens of FineWeb-Edu validation documents, KL to the teacher on 1.3M validation tokens, and
# zero-shot HellaSwag, ARC-Easy and PIQA through evals.py, unchanged.
# usage: python eval_students.py <tag> <weights.pt>
import sys, json, time, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sizes import S15
from snapshots import load, val_loss
from data import ShardLoader
from distill import kd_loss
from evals import OurLM, fineweb_val_docs, bits_per_byte, load_task, score_task, summarize
tag, path = sys.argv[1], sys.argv[2]
t0 = time.time(); out = {"tag": tag, "path": path}
model = load(path, cfg=S15)
out["val_loss_full"], out["val_tokens"] = val_loss(model)
teacher = load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt')
L, kl = ShardLoader("val", B=8, T=1024), 0.0
with torch.no_grad():
    for i in range(160):
        x, y = (t.cuda() for t in L.batch(i))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, k, _ = kd_loss(model(x), teacher(x), y)
        kl += k.item() / 160
out["kl_to_teacher"] = kl
del teacher
lm = OurLM(path, cfg=S15)
out["bpb"] = bits_per_byte(lm, fineweb_val_docs())
out["bench"] = {t: summarize(score_task(lm, load_task(t))) for t in ("hellaswag", "arc_easy", "piqa")}
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
json.dump(out, open(f'/home/amit/.cache/vidyalai-scratch/t13/post/student_{tag}.json', 'w'), indent=1)
