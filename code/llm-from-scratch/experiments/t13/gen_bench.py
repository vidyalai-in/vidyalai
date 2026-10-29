# Teacher generation throughput: batch size and KV-cache type, on the same 64 prompts (max 256 new tokens).
import sys, json, time, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from distill import teacher_generate
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
items = json.load(open('/home/amit/.cache/vidyalai-scratch/t13/data/prompts.json'))["train"][512:576]
convs = [it["prompt"] for it in items]
res = []
for cache, bs in [(None, 16), (None, 32), (None, 64), ("static", 32)]:
    extra = {"cache_implementation": cache} if cache else {}
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    if cache:  # warm-up: compilation happens on the first call
        teacher_generate(model, tok, convs[:bs], max_new=256, batch_size=bs, **extra)
    torch.cuda.synchronize(); t0 = time.time()
    out = teacher_generate(model, tok, convs, max_new=256, batch_size=bs, **extra)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = sum(o["new_tokens"] for o in out)
    r = {"cache": cache or "dynamic", "batch": bs, "seconds": round(dt, 1), "new_tokens": new, "tok_s": round(new / dt),
         "finished": sum(o["finished"] for o in out), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open('/home/amit/.cache/vidyalai-scratch/t13/post/gen_bench.json', 'w'), indent=1)
