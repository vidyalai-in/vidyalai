# Cost of logit distillation per micro-batch: teacher forward alone, student CE step, student KD step.
# usage: python bench_kd.py <student size> <micro-batch> [n]
import sys, time, json, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from data import ShardLoader
from model import GPT
from snapshots import load
from train import CONFIGS
from distill import kd_loss, topk_mass
size, B = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
torch.manual_seed(0)
teacher = torch.compile(load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt').requires_grad_(False))
student_raw = GPT(CONFIGS[size]).cuda(); student = torch.compile(student_raw)
loss_fn = torch.compile(kd_loss)
L = ShardLoader('train', B=B, T=1024)
ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
def timeit(fn, n):
    for i in range(3): fn(i)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n / 1000
batches = [tuple(t.cuda() for t in L.batch(i)) for i in range(8)]
def t_fwd(i):
    x, _ = batches[i % 8]
    with torch.no_grad(), ac(): teacher(x)
def s_ce(i):
    x, y = batches[i % 8]
    with ac(): loss = student(x, y)
    loss.backward()
def s_kd(i):
    x, y = batches[i % 8]
    with torch.no_grad(), ac(): tl = teacher(x)
    with ac(): loss, kl, ce = loss_fn(student(x), tl, y, 1.0, 1.0)
    loss.backward()
res = {"size": size, "B": B, "tokens": B * 1024}
torch.cuda.reset_peak_memory_stats()
res["teacher_fwd_s"] = timeit(t_fwd, n)
res["student_ce_s"] = timeit(s_ce, n); res["mem_ce_gib"] = torch.cuda.max_memory_allocated() / 2**30
torch.cuda.reset_peak_memory_stats()
res["student_kd_s"] = timeit(s_kd, n); res["mem_kd_gib"] = torch.cuda.max_memory_allocated() / 2**30
for k in ("teacher_fwd_s", "student_ce_s", "student_kd_s"):
    res[k.replace("_s", "_tok_s")] = B * 1024 / res[k]
# how much of the teacher's probability its top-k tokens hold (8 micro-batches)
with torch.no_grad(), ac():
    m = [topk_mass(teacher(batches[i][0])) for i in range(8)]
res["topk_mass"] = {k: sum(d[k] for d in m) / len(m) for k in m[0]}
print(json.dumps(res, indent=1))
json.dump(res, open(f'/home/amit/.cache/vidyalai-scratch/t13/post/bench_kd_{size}_B{B}.json', 'w'), indent=1)
