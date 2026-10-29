"""FSDP2 (fully_shard) on two CPU processes with the gloo backend: what each rank stores and sends (Part 14).

This shows correctness and memory per rank, not speed: two CPU processes stand in for two GPUs.
Rank 0 also trains an unsharded copy on the full batch, and the losses and final weights are compared.
    CUDA_VISIBLE_DEVICES= torchrun --nproc_per_node 2 fsdp_cpu.py
"""
import copy
import json

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from data import ShardLoader
from model import GPT, SMALL

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)
model = GPT(SMALL)                                      # identical initial weights on every rank
ref = copy.deepcopy(model) if rank == 0 else None
mesh = init_device_mesh("cpu", (world,))
for block in model.blocks:
    fully_shard(block, mesh=mesh)                       # one all-gather group per block, bottom-up
fully_shard(model, mesh=mesh)                           # the rest: embedding (tied output layer), final norm

sent = {"all_gather": 0, "reduce_scatter": 0}
_ag, _rs = dist.all_gather_into_tensor, dist.reduce_scatter_tensor


def all_gather(*a, **k):
    inp = k.get("input_tensor", a[1] if len(a) > 1 else None)
    sent["all_gather"] += inp.numel() * inp.element_size() * (world - 1)      # my shard goes to each other rank
    return _ag(*a, **k)


def reduce_scatter(*a, **k):
    inp = k.get("input", a[1] if len(a) > 1 else None)
    sent["reduce_scatter"] += inp.numel() * inp.element_size() * (world - 1) // world
    return _rs(*a, **k)


dist.all_gather_into_tensor, dist.reduce_scatter_tensor = all_gather, reduce_scatter

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
x, y = ShardLoader("train", B=2 * world, T=128).batch(0)          # global batch: 4 sequences of 128 tokens
mine = slice(rank * 2, rank * 2 + 2)                              # each rank trains on its 2 sequences
losses = []
for step in range(3):
    loss = model(x[mine], y[mine])
    loss.backward()                                   # gradients are averaged across ranks by reduce-scatter
    opt.step()
    opt.zero_grad()
    l = loss.detach().clone()
    dist.all_reduce(l)
    losses.append(l.item() / world)
    if step == 0:
        sent_step1 = dict(sent)

n_total = sum(p.numel() for p in model.parameters())
n_local = sum(p.to_local().numel() for p in model.parameters())
opt_local = sum(s.to_local().numel() * s.element_size() for st in opt.state.values() for s in st.values()
                if hasattr(s, "to_local") and s.dim() > 0)
full = {k: v.full_tensor() for k, v in model.state_dict().items()}   # all-gathers every shard
report = {"rank": rank, "params_total": n_total, "params_local": n_local, "optimizer_bytes_local": opt_local,
          "sent_step1_bytes": sent_step1, "losses": losses}

if rank == 0:
    ropt = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.0)
    ref_losses = []
    for step in range(3):
        loss = ref(x, y)                              # the whole batch, one process, no sharding
        loss.backward()
        ropt.step()
        ropt.zero_grad()
        ref_losses.append(loss.item())
    diff = max((full[k] - v).abs().max().item() for k, v in ref.state_dict().items())
    report.update({"ref_losses": ref_losses, "max_weight_diff": diff})
print(json.dumps(report), flush=True)
dist.destroy_process_group()
