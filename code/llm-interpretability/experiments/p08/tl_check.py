# Build Gemma 2 2B as a HookedTransformer on the GPU (lab/attribution.load_gemma_tl) and compare
# its last-position logits with plain transformers (runs/hf_ref.pt).
import sys, time, resource, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.attribution import load_gemma_tl
torch.set_grad_enabled(False)
t = time.time()
m = load_gemma_tl(transcoders=None, n_ctx=256)
print(f"load {time.time()-t:.1f}s GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB")
ref = torch.load("/home/amit/.cache/vidyalai-scratch/p08/runs/hf_ref.pt")
for p, r in ref.items():
    ids = torch.tensor([r["ids"]], device="cuda")
    assert m.to_tokens(p)[0].tolist() == r["ids"], (m.to_tokens(p)[0].tolist(), r["ids"])
    lg = m(ids)[0, -1].float().cpu()
    d = (lg - r["logits"]).abs()
    top = lg.topk(5)
    print(p, [(m.to_string(i), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
          f"max|dlogit| {d.max():.3f} mean {d.mean():.4f} top1 agree {lg.argmax()==r['logits'].argmax()}")
print(f"GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB host peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GiB")
