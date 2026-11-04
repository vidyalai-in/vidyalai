# Experiment 2: generate images for the guidance, sampler/step and negative-prompt sweeps; resumable.
# Usage: python exp2_generate.py <sweep> [--budget_s 660]
#   sweep "guidance": dpmpp2m, 20 steps, g in GS, prompts[:NP] x seeds[:NS]
#   sweep "steps":    ddim/euler/dpmpp2m at STEPS, g 7.5, prompts[:NP2] x 1 seed; plus a 100-step dpmpp2m reference
#   sweep "negative": NEG_TESTS x 4 seeds, with and without the negative prompt, dpmpp2m 20 steps g 7.5
# Writes runs/<sweep>.jsonl (one record per image) and runs/img/<sweep>/<key>.png; CLIP-B/32 embeddings to runs/emb/.
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import sd
from prompts import PROMPTS, NEG_TESTS

ap = argparse.ArgumentParser(); ap.add_argument("sweep"); ap.add_argument("--budget_s", type=float, default=660)
ap.add_argument("--np", type=int, default=24); ap.add_argument("--ns", type=int, default=3)
ap.add_argument("--np2", type=int, default=16)
a = ap.parse_args()
T0 = time.time()
GS = [1.0, 3.0, 5.0, 7.5, 12.0, 20.0]
STEPS = [4, 8, 15, 30]
SEEDS = [11, 22, 33, 44]


def jobs():
    if a.sweep == "guidance":
        for s in SEEDS[:a.ns]:
            for g in GS:
                for i in range(a.np):
                    yield dict(p=i, prompt=PROMPTS[i], neg="", seed=s, sampler="dpmpp2m", steps=20, g=g)
    elif a.sweep == "steps":
        for i in range(a.np2):
            yield dict(p=i, prompt=PROMPTS[i], neg="", seed=SEEDS[0], sampler="dpmpp2m", steps=100, g=7.5)
        for smp in ["dpmpp2m", "ddim", "euler"]:
            for st in STEPS:
                for i in range(a.np2):
                    yield dict(p=i, prompt=PROMPTS[i], neg="", seed=SEEDS[0], sampler=smp, steps=st, g=7.5)
    elif a.sweep == "negative":
        for use in (False, True):
            for s in SEEDS:
                for i, (pr, neg, concept) in enumerate(NEG_TESTS):
                    yield dict(p=i, prompt=pr, neg=neg if use else "", concept=concept, seed=s,
                               sampler="dpmpp2m", steps=20, g=7.5)


def key(j):
    return f"p{j['p']:02d}_s{j['seed']}_{j['sampler']}_{j['steps']}_g{j['g']}" + ("_neg" if j["neg"] else "")


out = f"runs/{a.sweep}.jsonl"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
todo = [j for j in jobs() if key(j) not in done]
print(f"{len(done)} done, {len(todo)} to do", flush=True)
if not todo:
    sys.exit(0)
os.makedirs(f"runs/img/{a.sweep}", exist_ok=True); os.makedirs("runs/emb", exist_ok=True)
pipe = sd.load_pipe()
judge = sd.ClipJudge()
f = open(out, "a")
n = 0
# group jobs that share sampler, steps, guidance and negative-ness into batches of up to BATCH images
BATCH = 4
groups, cur = [], []
for j in todo:
    if cur and (len(cur) == BATCH or (j["sampler"], j["steps"], j["g"], bool(j["neg"])) !=
                (cur[0]["sampler"], cur[0]["steps"], cur[0]["g"], bool(cur[0]["neg"]))):
        groups.append(cur); cur = []
    cur.append(j)
if cur:
    groups.append(cur)
with sd.GpuMonitor() as mon:
    for grp in groups:
        if time.time() - T0 > a.budget_s:
            break
        j0 = grp[0]
        sd.set_sampler(pipe, j0["sampler"])
        torch.cuda.synchronize(); t1 = time.perf_counter()
        r = pipe([j["prompt"] for j in grp], negative_prompt=[j["neg"] for j in grp] if j0["neg"] else None,
                 num_inference_steps=j0["steps"], guidance_scale=j0["g"],
                 generator=[torch.Generator("cuda").manual_seed(j["seed"]) for j in grp])
        torch.cuda.synchronize(); dt = (time.perf_counter() - t1) / len(grp)
        flags_sc, margins = sd.safety(pipe, r.images)
        embs = judge.images(r.images)
        txts = judge.texts([j["prompt"] for j in grp])
        css = sd.clip_score(embs, txts)
        for b, (j, im) in enumerate(zip(grp, r.images)):
            k = key(j)
            im.save(f"runs/img/{a.sweep}/{k}.png")
            np.save(f"runs/emb/{a.sweep}_{k}.npy", embs[b].numpy().astype(np.float16))
            rec = dict(j, key=k, nsfw=bool(r.nsfw_content_detected[b]), safety_margin=margins[b],
                       clip=float(css[b]), sec=dt, batch=len(grp))
            if a.sweep == "negative":
                ct = judge.texts([f"a photo of {j['concept']}"])
                rec["concept_sim"] = float((embs[b:b+1] * ct).sum())
            f.write(json.dumps(rec) + "\n"); n += 1
        f.flush()
print(f"wrote {n}; gpu median clock {mon.clock} MHz power {mon.power} W temp {mon.temp} C", flush=True)
json.dump({"n": n, "clock": mon.clock, "power": mon.power, "temp": mon.temp, "t": time.time()},
          open(f"runs/{a.sweep}_mon_{int(time.time())}.json", "w"))
