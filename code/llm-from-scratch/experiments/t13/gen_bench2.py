# Folded-GQA attention vs stock sdpa: same greedy output? and throughput on the gen_bench prompts.
import sys, json, time, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from distill import teacher_generate, register_folded_gqa
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
items = json.load(open('/home/amit/.cache/vidyalai-scratch/t13/data/prompts.json'))["train"][512:576]
convs = [it["prompt"] for it in items]
res, greedy = [], {}
for impl in ("sdpa", register_folded_gqa()):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=impl).cuda().eval()
    greedy[impl] = [o["text"] for o in teacher_generate(model, tok, convs[:16], max_new=64, batch_size=16, do_sample=False)]
    for bs in ((32, 64) if impl != "sdpa" else ()):
        torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        out = teacher_generate(model, tok, convs, max_new=256, batch_size=bs)
        torch.cuda.synchronize(); dt = time.time() - t0
        new = sum(o["new_tokens"] for o in out)
        r = {"attn": impl, "batch": bs, "seconds": round(dt, 1), "new_tokens": new, "tok_s": round(new / dt),
             "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(json.dumps(r), flush=True); res.append(r)
    del model; torch.cuda.empty_cache()
same = sum(a == b for a, b in zip(greedy["sdpa"], greedy["sdpa_folded"]))
print("greedy identical:", same, "of 16")
for a, b in zip(greedy["sdpa"], greedy["sdpa_folded"]):
    if a != b: print("DIFF\n", a[:200], "\n", b[:200])
json.dump({"runs": res, "greedy_identical": same}, open('/home/amit/.cache/vidyalai-scratch/t13/post/gen_bench2.json', 'w'), indent=1)
