"""Greedy HumanEval completions from a small open instruct model (same instruction as the API runs)."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path
D, RUNS = "/home/amit/.cache/vidyalai-scratch/evals/data/", Path("/home/amit/.cache/vidyalai-scratch/e08/runs")
CODE_INSTR = ("Complete the following Python function. Reply with the complete function, including its "
              "signature and any imports it needs, in a single ```python code block, and nothing else.\n\n")
name = sys.argv[1]; bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
he = json.load(open(D + "e08-humaneval.json"))
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
prompts = [tok.apply_chat_template([{"role": "user", "content": CODE_INSTR + "```python\n" + t["prompt"] + "```"}],
                                   tokenize=False, add_generation_prompt=True) for t in he]
order = sorted(range(len(he)), key=lambda i: len(prompts[i]))
out = {}
t0 = time.time()
for j in range(0, len(order), bs):
    idx = order[j:j + bs]
    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=768, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, row in zip(idx, g[:, enc["input_ids"].shape[1]:]):
        ids = row.tolist()
        hit_eos = tok.eos_token_id in ids or tok.pad_token_id in ids
        out[i] = dict(text=tok.decode(row, skip_special_tokens=True), truncated=not hit_eos)
    print(j, round(time.time() - t0, 1), flush=True)
tag = name.split("/")[-1]
with open(RUNS / f"he_{tag}.jsonl", "w") as f:
    for i in range(len(he)):
        f.write(json.dumps(dict(id=f"he{i:03d}-s0", **out[i])) + "\n")
print("seconds", round(time.time() - t0, 1), "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
