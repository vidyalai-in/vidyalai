"""Run a local VLM on the synthetic task set. Resumable: skips (item, size, setting) rows already in the output.

usage: run_local.py MODEL_ID OUT.jsonl --sizes 224 448 896 [--longest-edge N] [--limit N] [--time]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vlm

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--sizes", type=int, nargs="+", default=list(vlm.SIZES))
ap.add_argument("--longest-edge", type=int, default=None)     # SmolVLM processor setting
ap.add_argument("--limit", type=int, default=100)              # items per task
ap.add_argument("--time", action="store_true")                 # timing split instead of accuracy
ap.add_argument("--budget-s", type=float, default=780)         # stop cleanly before ~13 min
args = ap.parse_args()

t_start = time.time()
setting = f"le{args.longest_edge}" if args.longest_edge else "default"
if not args.time and os.path.exists(args.out):
    have = {(json.loads(l)["id"], json.loads(l)["size"], json.loads(l)["setting"]) for l in open(args.out)}
    want = {(i.id, s, setting) for i in vlm.make_items(150) if int(i.id[-3:]) < args.limit for s in args.sizes}
    if want <= have:
        print("nothing to do"); sys.exit(0)
m = vlm.VLM(args.model, image_size=args.longest_edge)
print("loaded", args.model, round(time.time() - t_start, 1), "s", flush=True)
items = [i for i in vlm.make_items(150) if int(i.id[-3:]) < args.limit]

if args.time:
    rows = []
    probe = [i for i in items if i.task == "chart"][0]
    for size in args.sizes:
        img = vlm.render(probe, size)
        q = probe.question + " " + vlm.SUFFIX[probe.task]
        with vlm.GpuSampler() as g:
            r = m.time_split(img, q, warmup=3, reps=10)
        r.update(dict(model=args.model, size=size, setting=setting, gpu=g.stop()))
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("_all")}), flush=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sys.exit(0)

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["id"], r["size"], r["setting"]))
n_new = 0
todo = [(size, it) for size in args.sizes for it in items if (it.id, size, setting) not in done]
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(6)   # CPU preprocessing (render, resize, tile) runs ahead of the GPU
prep = lambda job: m.inputs(vlm.render(job[1], job[0]), job[1].question + " " + vlm.SUFFIX[job[1].task], device=None)
AHEAD = 12                     # bounded: one SmolVLM item is ~53 MB of fp32 pixels
futs = {k: pool.submit(prep, todo[k]) for k in range(min(AHEAD, len(todo)))}
with open(args.out, "a") as f, vlm.GpuSampler(period=1.0) as g:
    for k, (size, it) in enumerate(todo):
            fut = futs.pop(k)
            if k + AHEAD < len(todo):
                futs[k + AHEAD] = pool.submit(prep, todo[k + AHEAD])
            if time.time() - t_start > args.budget_s:
                print("budget reached; resume later", flush=True)
                break
            t0 = time.time()
            text, info = m.ask(None, None, prepped=fut.result())
            sc = vlm.score(it, text)
            f.write(json.dumps(dict(id=it.id, task=it.task, size=size, setting=setting, model=args.model,
                                    answer=it.answer, text=text, wall_s=round(time.time() - t0, 3), **info, **sc)) + "\n")
            n_new += 1
            if n_new % 100 == 0:
                f.flush()
                print(size, it.id, n_new, round(time.time() - t_start), "s", flush=True)
pool.shutdown(cancel_futures=True)
print("gpu", g.stop(), "new rows", n_new, "total s", round(time.time() - t_start), flush=True)
