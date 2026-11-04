"""XSTest replies from an open instruct model: greedy, chat template, no system prompt, 256 new tokens, bf16 on the GPU."""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals")
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from lab import safety_evals as S
name, tag = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
rows = S.load_xstest()
t0 = time.time()
outs = S.generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=256, batch_size=int(sys.argv[3]) if len(sys.argv) > 3 else 16)
dt = time.time() - t0
with open(f"/home/amit/.cache/vidyalai-scratch/e10/runs/xstest_{tag}.jsonl", "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": f"r0-{r['id']}", "model": name, "text": o}) + "\n")
print(json.dumps({"model": name, "seconds": round(dt, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name()}))
