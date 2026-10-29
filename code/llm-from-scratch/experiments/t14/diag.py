"""Where a large configuration runs out of memory: allocated memory after each phase of the first step."""
import sys, torch
from data import ShardLoader
from memory import checkpoint_blocks
from model import GPT, GPTConfig
d, L, comp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "compile"
torch.cuda.set_per_process_memory_fraction(7.45 * 2**30 / torch.cuda.mem_get_info()[1])
cfg = GPTConfig(d_model=d, n_layer=L, n_head=d // 64, n_kv_head=d // 192, d_ff=8 * d // 3)
G = 2**30
m = GPT(cfg).cuda().to(torch.bfloat16)
checkpoint_blocks(m)
run = torch.compile(m) if comp else m
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
x, y = (t.cuda() for t in ShardLoader("train", B=4, T=1024).batch(0))
def p(s): print(f"{s:12} alloc {torch.cuda.memory_allocated()/G:.2f}  peak {torch.cuda.max_memory_allocated()/G:.2f}", flush=True)
p("weights")
try:
    for i in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = run(x, y)
        p(f"fwd{i}")
        loss.backward(); p(f"bwd{i}")
        opt.step(); p(f"opt{i}"); opt.zero_grad(set_to_none=True)
except torch.OutOfMemoryError as e:
    p("OOM"); print(str(e)[:200])
