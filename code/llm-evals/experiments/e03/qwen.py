"""Part 3: Qwen2.5-1.5B-Instruct on MMLU-1140, same system and user prompt as the Claude runs,
scored by the next-token probabilities of the letters A-D after the assistant turn starts.
Writes runs/qwen-1.5b.jsonl: per item the four letter probabilities (renormalized over A-D),
the argmax letter and whether it is right. Deterministic: no sampling."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from run import ITEMS, LETTERS, SYSTEM, prompt  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path(__file__).parent / "runs" / "qwen-1.5b.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
assert all(len(tok.encode(L, add_special_tokens=False)) == 1 for L in LETTERS)

texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": prompt(it)}],
                                 tokenize=False, add_generation_prompt=True) for it in ITEMS]
t0 = time.time()
rows = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        full = torch.softmax(logits, -1)
        p4 = full[:, letter_ids]
        for j, it in enumerate(ITEMS[i:i + 16]):
            p = (p4[j] / p4[j].sum()).tolist()
            pred = LETTERS[int(torch.argmax(p4[j]))]
            rows.append(dict(id=it["id"], subject=it["subject"], probs=p, mass_abcd=float(p4[j].sum()),
                             pred=pred, answer=LETTERS[it["answer"]], correct=int(pred == LETTERS[it["answer"]]),
                             p_correct=p[it["answer"]]))
OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
acc = sum(r["correct"] for r in rows) / len(rows)
print(f"{len(rows)} items, accuracy {acc:.4f}, {time.time() - t0:.1f} s, "
      f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, torch {torch.__version__}")
