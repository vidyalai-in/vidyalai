# Greedy replies from the PyTorch chat model (FP32 and BF16), to compare with llama.cpp's.
import json, sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from sft import chat_tokenizer, prompt_ids, END
from snapshots import load
PROMPTS = ["What is the capital of France?", "Explain photosynthesis to a ten-year-old in two sentences.",
           "Give me three tips for sleeping better.", "What is 17 + 25?",
           "Write a haiku about the monsoon.", "Why is the sky blue?"]
tok, model = chat_tokenizer(), load("/home/amit/.cache/vidyalai-scratch/t11/runs/onpolicy/model_final.pt")
out = []
for p in PROMPTS:
    idx = torch.tensor([prompt_ids([{"role": "user", "content": p}], tok)], device="cuda")
    for bf16 in (False, True):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            new = model.generate(idx, 256, temperature=0, eos_id=END)[0, idx.shape[1]:].tolist()
        ended = new[-1] == END
        out.append({"prompt": p, "bf16": bf16, "ids": new, "ended": ended,
                    "text": tok.decode(new[:-1] if ended else new)})
        print(json.dumps(out[-1]["text"])[:300], ended, len(new), "bf16" if bf16 else "fp32")
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/t15/post/chat_reference.json", "w"), indent=1)
