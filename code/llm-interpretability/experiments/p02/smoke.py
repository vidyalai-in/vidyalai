import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import lens as LL

name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
key = {"gpt2": "gpt2", "EleutherAI/pythia-160m": "pythia-160m", "Qwen/Qwen2.5-0.5B": "qwen2.5-0.5b"}[name]
dtype = torch.float32
m = LL.load(name, dtype=dtype)
d = torch.load(f"/home/amit/.cache/vidyalai-scratch/p02/runs/tokens-{key}.pt")
tok = d["train"][:8].cuda()
logits, H = LL.resid_points(m, tok)
print("points", H.shape, "final-point lens == logits, max abs diff:",
      (LL.logit_lens(m, H[-1]) - logits).abs().max().item())
batches = iter(d["train"][i:i + 8] for i in range(0, 8 * 30, 8))
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
lens, log = LL.train_tuned_lens(m, batches, kind="full", opt="muon", steps=10, log_every=5)
torch.cuda.synchronize()
print(f"10 steps x 4096 tokens: {time.time()-t:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
t = time.time()
r = LL.evaluate(m, d["eval"][:16], {"tuned": lens})
torch.cuda.synchronize(); print(f"eval 16 chunks: {time.time()-t:.1f}s")
import numpy as np
print("logit kl by point", np.round(np.mean(r["logit"]["kl"], 0), 2))
print("tuned kl by point", np.round(np.mean(r["tuned"]["kl"], 0), 2))
print("final ce", np.mean(r["final_ce"]))
