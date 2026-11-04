"""Score one open model on all 5,700 MMLU-Redux items: zero-shot, log-probability of the letters
' A'..' D' after 'Answer:'. bf16 on the GPU, one forward pass per item. Saves (5700, 4) log-probs."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import items
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[1]
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
data = items.load_redux()

def prompt(it):
    subj = it["subject"].replace("_", " ")
    opts = "\n".join(f"{L}. {c}" for L, c in zip(items.LETTERS, it["choices"]))
    return (f"The following are multiple choice questions (with answers) about {subj}.\n\n"
            f"{it['question'].strip()}\n{opts}\nAnswer:")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
# Sort by length so batches pad little; restore order afterwards.
ps = [prompt(it) for it in data]
order = np.argsort([len(p) for p in ps])
lp = items.letter_logprobs(model, tok, [ps[i] for i in order], batch_size=bs)
out = np.empty_like(lp); out[order] = lp
tag = name.split("/")[-1]
np.save(f"/home/amit/.cache/vidyalai-scratch/e02/runs/open-{tag}.npy", out)
gold = np.array([it["answer"] for it in data])
acc = (out.argmax(1) == gold).mean()
info = dict(model=name, items=len(data), acc_vs_key=float(acc), seconds=round(time.time() - t0, 1),
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20))
json.dump(info, open(f"/home/amit/.cache/vidyalai-scratch/e02/runs/open-{tag}.json", "w"))
print(info)
