# Where a generate() step's time goes: torch.profiler over 32 new tokens for 32 prompts.
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
print("attn impl", model.config._attn_implementation)
items = json.load(open('/home/amit/.cache/vidyalai-scratch/t13/data/prompts.json'))["train"][512:544]
P = [tok.apply_chat_template(it["prompt"], tokenize=False, add_generation_prompt=True) for it in items]
enc = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
print("prompt shape", enc["input_ids"].shape)
model.generate(**enc, max_new_tokens=8, pad_token_id=tok.pad_token_id)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.generate(**enc, max_new_tokens=32, min_new_tokens=32, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
