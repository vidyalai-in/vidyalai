"""Train SAEs on GPT-2 small blocks.8.hook_resid_pre (TransformerLens from_pretrained: fp32, LayerNorm folded,
writing weights centered), FineWeb-Edu windows of 128 tokens.
usage: train_gpt2.py <config.json> ; config: {"name", "steps", "saes": {name: {kind, n, k|l1, lr}}, "time_limit"}
Resumes from runs/<name>.ckpt if present.
"""
import sys, json, os, time
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
cfg = json.load(open(sys.argv[1]))
R = "/home/amit/.cache/vidyalai-scratch/p07/runs"
ckpt = f"{R}/{cfg['name']}.ckpt"
data = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gpt2-tokens.pt")
model = HookedTransformer.from_pretrained("gpt2", device="cuda")
for p in model.parameters():
    p.requires_grad_(False)
LAYER = 8

state = torch.load(ckpt) if os.path.exists(ckpt) else None
start_chunk = state["chunk"] if state else 0
chunks = S.gpt2_chunks(model, data["train"], LAYER, seqs_per_chunk=2048, start=start_chunk)
saes = {}
first = None
if state is None:
    c, first = next(chunks)
    scale = S.estimate_scale(first)
    print("scale", scale, flush=True)
for k, c_ in cfg["saes"].items():
    s = S.SAE(768, c_["n"], kind=c_["kind"], k=c_.get("k"), l1=c_.get("l1", 0.0), seed=c_.get("seed", 0)).cuda()
    if state:
        s.load_state_dict(state["sae"][k])
    else:
        s.scale.fill_(scale)
        if s.kind == "topk":                       # b_pre starts at the data mean (Gao et al. use the geometric median)
            s.b_dec.data = (first[:65536].mean(0) / scale).clone()
    saes[k] = s


def stream():
    global first
    if first is not None:
        f, first = first, None
        yield 0, f
        del f
    yield from chunks


torch.set_grad_enabled(True)
t0 = time.time()
with open(f"{R}/{cfg['name']}.log.jsonl", "a") as log:
    step = S.train(saes, {k: v["lr"] for k, v in cfg["saes"].items()}, stream(), cfg["steps"], ckpt=ckpt, log=log,
                   state=state, time_limit=cfg.get("time_limit"))
print("stopped at step", step, "of", cfg["steps"], "wall", round(time.time() - t0, 1), "s",
      "peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
torch.save({"sae": {k: s.state_dict() for k, s in saes.items()}, "cfg": cfg, "step": step}, f"{R}/{cfg['name']}.saes.pt")
