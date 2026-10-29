# Teacher replies from Qwen2.5-1.5B-Instruct (Apache-2.0) to Part 10's prompts, with distill.teacher_generate.
# usage: python gen_teacher.py <split> <start> <count> <batch> <max_new> <out.jsonl> [static]
# Appends one JSON line per prompt; skips indices already in the file, so a crash resumes.
import sys, json, os, time, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from distill import teacher_generate, register_folded_gqa
from transformers import AutoModelForCausalLM, AutoTokenizer
split, start, count, bs, max_new, path = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
static = len(sys.argv) > 7 and sys.argv[7] == "static"
name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
impl = sys.argv[8] if len(sys.argv) > 8 else register_folded_gqa()     # the first 512 train replies used stock "sdpa"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=impl).cuda().eval()
items = json.load(open('/home/amit/.cache/vidyalai-scratch/t13/data/prompts.json'))[split][start:start + count]
done = {json.loads(l)["i"] for l in open(path)} if os.path.exists(path) else set()
items = [it for it in items if it["i"] not in done]
print("to do", len(items), "static" if static else "dynamic", flush=True)
extra = {"cache_implementation": "static"} if static else {}
torch.manual_seed(1234)
f = open(path, "a")
chunk = bs * 8                                          # write every 8 batches
for c in range(0, len(items), chunk):
    part = items[c:c + chunk]
    torch.cuda.synchronize(); t0 = time.time()
    res = teacher_generate(model, tok, [it["prompt"] for it in part], max_new=max_new, batch_size=bs, **extra)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = sum(r["new_tokens"] for r in res)
    for it, r in zip(part, res):
        f.write(json.dumps({"i": it["i"], "split": split, "source": it["source"], **r}) + "\n")
    f.flush()
    print(json.dumps({"done": c + len(part), "n": len(part), "seconds": round(dt, 1), "new_tokens": new,
                      "tok_s": round(new / dt), "finished": sum(r["finished"] for r in res),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
