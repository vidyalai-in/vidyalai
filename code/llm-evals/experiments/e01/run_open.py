"""Score one open model on e01's item sets by letter log-probability (zero-shot, raw text prompt,
bf16 on the GPU). Usage: run_open.py <hf model id>. Writes runs/open-<name>.json."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import validity as V
D = "/home/amit/.cache/vidyalai-scratch/evals/data"
SETS = {"arce": "e01-arc-easy.json", "arcc": "e01-arc-challenge.json", "mmlu": "mmlu-1140.json",
        "pro": "e01-mmlu-pro-840.json"}
mid = sys.argv[1]
t0 = time.time()
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
out = {"model": mid, "torch": torch.__version__, "preds": {}}
for s, f in SETS.items():
    items = V.load(f"{D}/{f}")
    kinds = ["full", "choices_only"] if s in ("arcc", "mmlu", "pro") else ["full"]
    for kind in kinds:
        bs = 4 if s == "pro" else 8
        p = V.score_letters(model, tok, items, kind, batch_size=bs)
        acc = sum(int(a == it["answer"]) for a, it in zip(p, items)) / len(items)
        out["preds"][f"{s}-{kind}"] = p
        print(mid, s, kind, len(items), round(acc, 4), round(time.time() - t0), flush=True)
out["seconds"] = round(time.time() - t0)
out["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/e01/runs/open-{mid.split('/')[-1]}.json", "w"))
print("done", out["seconds"], "s", out["max_mem_gb"], "GB")
