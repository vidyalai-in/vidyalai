"""What one eager BF16 micro-batch of model.py saves for backward, tensor by tensor (Part 14).

Pack hooks see every tensor autograd saves; storages are de-duplicated, so a tensor saved by two ops
counts once. The module that was running when it was saved and the op that produced it label each one.
Also checks the hook total against torch.cuda.memory_allocated() before and after the forward pass,
and compares with memory.activation_bytes() and Korthikanti's formula.
    python saved.py --mb 2
"""
import argparse
import collections
import json

import torch

from data import ShardLoader
from memory import activation_bytes, korthikanti
from model import GPT, MAIN

ap = argparse.ArgumentParser()
ap.add_argument("--mb", type=int, default=2)
ap.add_argument("--out", default="/home/amit/.cache/vidyalai-scratch/t14/post/saved_eager.json")
a = ap.parse_args()

torch.manual_seed(0)
model = GPT(MAIN).cuda()
x, y = (t.cuda() for t in ShardLoader("val", B=a.mb, T=1024).batch(0))
current = ["top"]
for name, m in model.named_modules():
    if name.count(".") <= 2 and name:
        m.register_forward_pre_hook(lambda mod, inp, n=name: current.__setitem__(0, n))
seen, rows = set(), collections.Counter()


def pack(t):
    if t.is_cuda and not isinstance(t, torch.nn.Parameter):        # FP32 weights are counted as weights
        st = t.untyped_storage()
        if st.data_ptr() not in seen:
            seen.add(st.data_ptr())
            mod = current[0]
            where = ("blocks.N." + ".".join(mod.split(".")[2:])) if mod.startswith("blocks.") else mod
            prod = type(t.grad_fn).__name__ if t.grad_fn else "no grad_fn"
            rows[(where, prod, str(t.dtype).replace("torch.", ""), tuple(t.shape[-1:]))] += st.nbytes()
    return t


# warm-up so cuBLAS/flash workspaces exist before measuring
with torch.autocast("cuda", dtype=torch.bfloat16):
    model(x, y).backward()
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
before = torch.cuda.memory_allocated()
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
torch.cuda.synchronize()
after = torch.cuda.memory_allocated()
tokens = a.mb * 1024
hook_total = sum(rows.values())
print(f"micro-batch {a.mb} x 1024: allocated grew {(after - before) / 2**20:.1f} MiB; hooks saw {hook_total / 2**20:.1f} MiB")
per_layer = collections.Counter()
for (where, prod, dt, last), b in sorted(rows.items(), key=lambda kv: -kv[1]):
    if where.startswith("blocks.N"):
        per_layer[(where, prod, dt, last)] += b
    print(f"{where:24} {prod:28} {dt:9} {str(last):8} {b / 2**20:9.2f} MiB  {b / tokens / (MAIN.n_layer if where.startswith('blocks.N') else 1):9.0f} B/token{'/layer' if where.startswith('blocks.N') else ''}")
blk = sum(per_layer.values()) / tokens / MAIN.n_layer
pred = activation_bytes(MAIN, tokens)
print(f"blocks: measured {blk:.0f} bytes per token per layer; predicted {(pred['attention'] + pred['mlp']) / tokens / MAIN.n_layer:.0f}")
print(f"predicted total {sum(pred.values()) / 2**20:.1f} MiB  {pred}")
print(f"Korthikanti (a=12, h=768): {korthikanti(1024, a.mb, 768, 12) / tokens:.0f} bytes per token per layer; "
      f"without the 5as/h score term: {34 * 768:.0f}")
json.dump({"mb": a.mb, "tokens": tokens, "alloc_delta": after - before, "hook_total": hook_total,
           "rows": [[*k[:3], list(k[3]), v] for k, v in rows.items()], "pred": pred,
           "block_bytes_per_token_layer": blk}, open(a.out, "w"), indent=1)
