# Part 7: the same prompts, greedy and sampled, from snapshots across the run.
# usage (from fromscratch/): python samples.py <run dir> <step> [...]
import sys, json
from snapshots import load, complete
PROMPTS = ["The water cycle is", "In 1492, Christopher Columbus", "The best way to learn a new language is"]
run, steps = sys.argv[1], [int(s) for s in sys.argv[2:]]
out = []
for step in steps:
    model = load(f"{run}/model_{step:05d}.pt")
    for p in PROMPTS:
        g = complete(model, p, n_tokens=48)
        s = complete(model, p, n_tokens=48, temperature=0.8, top_k=50, seed=0)
        out.append({"step": step, "prompt": p, "greedy": g, "sampled": s})
        print(json.dumps(out[-1]), flush=True)
    del model
