# Which choices cost memory under torch.compile? Peak at one micro-batch size, for variants of the model.
import argparse, json, time, torch
from common import heldout_batch
import model as M

ap = argparse.ArgumentParser()
ap.add_argument("--mb", type=int, default=4); ap.add_argument("--keep_logits", type=int, default=0)
ap.add_argument("--rope_bf16", type=int, default=0); ap.add_argument("--compile", type=int, default=1)
ap.add_argument("--loss_only", type=int, default=0)
a = ap.parse_args()
if a.rope_bf16:
    def apply_rope(x, cos, sin):
        T = x.shape[-2]; cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    M.apply_rope = apply_rope
torch.manual_seed(1337)
model = M.GPT(M.MAIN).cuda()
if a.loss_only:
    class LossOnly(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x, y): return s.m(x, y)[1]
    step_model = LossOnly(model)
else:
    step_model = model
step_model = torch.compile(step_model) if a.compile else step_model
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
x, y = heldout_batch(a.mb, 1024)
def micro():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = step_model(x, y)
    if a.loss_only: loss = out
    elif a.keep_logits: logits, loss = out
    else: loss = out[1]; del out
    loss.backward()
for _ in range(2): micro()
opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
ts = []
for _ in range(8):
    torch.cuda.synchronize(); t = time.perf_counter(); micro(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
print("RESULT", json.dumps(dict(vars(a), peak=round(torch.cuda.max_memory_allocated() / 2**30, 3), med=round(sorted(ts[2:])[3], 4))))
