"""One training configuration: memory per phase, peak allocated/reserved, and tokens per second (Part 14).

Imports model.py, data.py, train.py's optimizer groups and memory.py without changing them.
    python bench.py --tag base16 --mb 16 --accum 4 --compile
    python bench.py --tag ckpt_all --mb 16 --ckpt all --compile
    python bench.py --tag big --d 1536 --L 24 --mb 4 --ckpt all --opt adamw8bit --compile
Writes t14/runs/<tag>.json; with --snapshot also a torch.cuda.memory snapshot of one timed step.
"""
import argparse
import json
import os
import sys

if "--expandable" in sys.argv:                     # must be set before CUDA initializes
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pickle
import resource
import statistics
import time

import torch

from data import ShardLoader
from memory import ChunkedLossGPT, CPUAdamW, add_lora, checkpoint_blocks
from model import GPT, GPTConfig

RUNS = "/home/amit/.cache/vidyalai-scratch/t14/runs"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--d", type=int, default=768)
ap.add_argument("--L", type=int, default=10)
ap.add_argument("--mb", type=int, default=16)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--steps", type=int, default=3, help="timed optimizer steps after one warm-up step")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--compile-blocks", action="store_true", help="compile each block on its own instead of the whole model")
ap.add_argument("--ckpt", choices=["none", "all", "every2", "every3", "selective"], default="none")
ap.add_argument("--opt", choices=["adamw", "adamw8bit", "cpu", "bf16pure", "bf16adam8"], default="adamw")
ap.add_argument("--gpu-bf16", action="store_true", help="keep the GPU copy of the weights in BF16 (with --opt cpu)")
ap.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 = full training")
ap.add_argument("--lora-bf16", action="store_true", help="store the frozen base weights in BF16")
ap.add_argument("--cap-gib", type=float, default=7.45, help="caching-allocator limit, so WSL2 can't spill to host memory")
ap.add_argument("--snapshot", action="store_true")
ap.add_argument("--chunked-loss", action="store_true", help="memory.ChunkedLossGPT: never hold the full logits")
ap.add_argument("--chunk", type=int, default=2048, help="tokens per loss chunk with --chunked-loss")
ap.add_argument("--expandable", action="store_true", help="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
ap.add_argument("--budget", type=float, default=None, help="torch.compile activation_memory_budget (0..1)")
a = ap.parse_args()

torch.manual_seed(1337)
torch.cuda.init()
free0, total = torch.cuda.mem_get_info()
torch.cuda.set_per_process_memory_fraction(a.cap_gib * 2**30 / total)
if a.budget is not None:
    torch._functorch.config.activation_memory_budget = a.budget
cfg = GPTConfig(d_model=a.d, n_layer=a.L, n_head=a.d // 64, n_kv_head=a.d // 192, d_ff=8 * a.d // 3)
rec = {"args": vars(a), "config": cfg.__dict__, "context_bytes": total - free0, "total_bytes": total,
       "torch": torch.__version__}
mem = torch.cuda.memory_allocated


def save():
    json.dump(rec, open(os.path.join(RUNS, a.tag + ".json"), "w"), indent=1)


try:
    ChunkedLossGPT.chunk = a.chunk
    model = (ChunkedLossGPT if a.chunked_loss else GPT)(cfg).cuda()
    rec["params"] = model.num_params()
    rec["after_weights"] = mem()
    if a.opt in ("bf16pure", "bf16adam8") or a.gpu_bf16 or (a.lora and a.lora_bf16):
        model.to(torch.bfloat16)
        rec["after_cast"] = mem()
    if a.lora:
        add_lora(model, r=a.lora)
        if a.lora_bf16:                                         # adapters stay FP32, base BF16
            for m in model.modules():
                if hasattr(m, "a"):
                    m.a.data, m.b.data = m.a.data.float(), m.b.data.float()
    rec["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if a.ckpt != "none":
        every = {"all": 1, "every2": 2, "every3": 3, "selective": 1}[a.ckpt]
        checkpoint_blocks(model, every=every, selective=a.ckpt == "selective")
    hp = argparse.Namespace(lr=6e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, fused=True)
    if a.opt in ("adamw8bit", "bf16adam8"):
        import bitsandbytes as bnb
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = bnb.optim.AdamW8bit([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                                  lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
        rec["bnb"] = bnb.__version__
    elif a.opt == "cpu":
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = CPUAdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                       lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    else:
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                                lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    if a.compile_blocks:
        for block in model.blocks:
            block.compile()                     # each block's backward returns only its own gradients
    run = torch.compile(model) if a.compile and not a.compile_blocks else model
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    batches = ShardLoader("train", B=a.mb, T=1024).batches(step=0, accum=a.accum)
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt_times = []

    def step(phases=None):
        loss_sum = torch.zeros((), device="cuda")
        for i in range(a.accum):
            x, y = (t.pin_memory().cuda(non_blocking=True) for t in next(batches))
            with autocast:
                loss = run(x, y) / a.accum
            if phases is not None:
                phases.append((f"fwd{i}", mem(), torch.cuda.max_memory_allocated()))
            loss.backward()
            if phases is not None:
                phases.append((f"bwd{i}", mem(), torch.cuda.max_memory_allocated()))
            loss_sum += loss.detach()
        torch.cuda.synchronize()
        t_opt = time.time()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if phases is not None:
            rec["grad_ptrs"] = {p.grad.untyped_storage().data_ptr(): p.grad.untyped_storage().nbytes() for p in trainable}
            phases.append(("clip", mem(), torch.cuda.max_memory_allocated()))
        opt.step()
        if phases is not None:
            phases.append(("opt", mem(), torch.cuda.max_memory_allocated()))
        opt.zero_grad(set_to_none=True)
        out = loss_sum.item()
        opt_times.append(time.time() - t_opt)
        return out

    t0 = time.time()
    warm_phases = []
    rec["warm_phases"] = warm_phases
    rec["warmup_loss"] = step(warm_phases)
    rec["warmup_s"] = round(time.time() - t0, 2)
    torch.cuda.synchronize()
    rec["after_warmup"] = mem()
    torch.cuda.reset_peak_memory_stats()
    if a.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    phases, times, losses = [("start", mem(), mem())], [], []
    for s in range(a.steps):
        torch.cuda.synchronize()
        t = time.time()
        losses.append(step(phases if s == 0 else None))
        times.append(time.time() - t)
        if s == 0 and a.snapshot:
            pickle.dump(torch.cuda.memory._snapshot(), open(os.path.join(RUNS, a.tag + ".snapshot.pickle"), "wb"))
            torch.cuda.memory._record_memory_history(enabled=None)
    st = torch.cuda.memory_stats()
    free1, _ = torch.cuda.mem_get_info()
    rec.update({"phases": phases, "step_s": times, "losses": losses, "opt_s": opt_times[1:],
                "tok_per_s": round(a.mb * 1024 * a.accum / statistics.median(times)),
                "peak_alloc": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
                "reserved_end": torch.cuda.memory_reserved(), "nonpytorch_end": total - free1 - torch.cuda.memory_reserved(),
                "inactive_split_peak": st.get("inactive_split_bytes.all.peak", 0),
                "alloc_retries": st.get("num_alloc_retries", 0), "oom": False})
    rec.pop("grad_ptrs") if not a.snapshot else None
except torch.OutOfMemoryError as e:
    import traceback
    rec.update({"oom": True, "error": str(e).splitlines()[0][:300], "peak_alloc": torch.cuda.max_memory_allocated(),
                "where": [ln for ln in traceback.format_exc().splitlines() if "bench.py" in ln or "memory.py" in ln][-3:]})
rec["host_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
save()
print(json.dumps({k: rec.get(k) for k in ("params", "trainable", "tok_per_s", "peak_alloc", "peak_reserved", "oom",
                                           "context_bytes", "nonpytorch_end", "warmup_s", "losses", "host_maxrss")}))
