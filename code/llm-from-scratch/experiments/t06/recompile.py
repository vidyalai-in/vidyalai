# Does train.py's validation pass (model.eval() + torch.no_grad()) recompile the compiled 88M model?
import sys, time
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from model import GPT, MAIN

torch.manual_seed(0)
raw = GPT(MAIN).cuda()
model = torch.compile(raw)
x = torch.randint(0, MAIN.vocab_size, (16, 1024), device="cuda")
def timed(label, fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    print(f"{label}: {time.perf_counter() - t:.2f} s", flush=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    timed("train fwd+bwd, first call", lambda: model(x, x).backward())
    timed("train fwd+bwd, second call", lambda: model(x, x).backward())
    model.eval()
    with torch.no_grad():
        timed("eval no_grad, first call", lambda: model(x, x))
        timed("eval no_grad, second call", lambda: model(x, x))
    model.train()
    timed("train again", lambda: model(x, x).backward())
