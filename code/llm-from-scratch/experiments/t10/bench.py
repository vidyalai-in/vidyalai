# Throughput of SFT micro-batches: eager vs torch.compile(dynamic=True) vs compile with fixed shapes.
import sys, time, json, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import Conversations, plan, collate
from snapshots import load
mode, budget = sys.argv[1], int(sys.argv[2])
data = Conversations("train", 20000)
groups = plan(data, 1, budget, 1337)[0][:40]
raw = load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt').train()
opt = torch.optim.AdamW(raw.parameters(), lr=1e-6, fused=True)
model = {"eager": raw, "dyn": torch.compile(raw, dynamic=True), "static": torch.compile(raw, dynamic=False)}[mode]
if mode == "static":
    torch._dynamo.config.cache_size_limit = 64
res = []
t_all = time.time()
for i, (g, T) in enumerate(groups):
    x, y = (t.cuda() for t in collate(data, g, T, "assistant"))
    torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = time.time() - t
    res.append((x.shape[0], T, dt, x.numel() / dt))
    print(i, x.shape[0], T, round(dt, 3), round(x.numel() / dt), flush=True)
steady = [r for r in res[10:] if r[2] < 5]
print(json.dumps({"mode": mode, "budget": budget, "total_s": round(time.time() - t_all, 1),
                  "median_tok_s": sorted(r[3] for r in steady)[len(steady) // 2],
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
