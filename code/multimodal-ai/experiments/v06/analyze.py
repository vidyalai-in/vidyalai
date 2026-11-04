# Summarize the v06 runs into runs/results.json (small; committed with the code).
# Intervals: cluster bootstrap over prompts (seeds averaged inside a prompt), 5,000 resamples, 95%.
import json, os, sys, itertools
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
from sd import wilson

R = {}
rng = np.random.default_rng(0)


def boot_prompts(by_prompt, n=5000):
    """by_prompt: list of per-prompt values (already averaged over seeds)."""
    v = np.asarray(by_prompt, float)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(name):
    p = f"runs/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def emb(sweep, key):
    return np.load(f"runs/emb/{sweep}_{key}.npy").astype(np.float32)


# ---------------- guidance sweep
G = load("guidance")
judge = {}
if os.path.exists("runs/judge.jsonl"):
    for l in open("runs/judge.jsonl"):
        r = json.loads(l); judge[r["key"]] = r
hand = {}
if os.path.exists("hand_labels.json"):
    hand = json.load(open("hand_labels.json"))
if G:
    R["guidance"] = {}
    gs = sorted({r["g"] for r in G})
    for g in gs:
        rows = [r for r in G if r["g"] == g and not r["nsfw"]]
        prompts = sorted({r["p"] for r in rows})
        clip_p, div_p, judge_p = [], [], []
        for p in prompts:
            pr = [r for r in rows if r["p"] == p]
            clip_p.append(np.mean([r["clip"] for r in pr]))
            if len(pr) >= 2:
                E = np.stack([emb("guidance", r["key"]) for r in pr]); E /= np.linalg.norm(E, axis=1, keepdims=True)
                div_p.append(np.mean([1 - E[i] @ E[j] for i, j in itertools.combinations(range(len(pr)), 2)]))
            js = [judge[r["key"]]["verdict"] == "yes" for r in pr if r["key"] in judge and judge[r["key"]].get("verdict")]
            if js:
                judge_p.append(np.mean(js))
        yes = [judge[r["key"]]["verdict"] == "yes" for r in rows if r["key"] in judge and judge[r["key"]].get("verdict")]
        R["guidance"][str(g)] = dict(
            n=len(rows), prompts=len(prompts), seeds=len({r["seed"] for r in rows}),
            nsfw=sum(r["nsfw"] for r in G if r["g"] == g),
            clip=boot_prompts(clip_p), diversity=boot_prompts(div_p) if div_p else None,
            judge_yes=boot_prompts(judge_p) if judge_p else None,
            judge_n=len(yes), judge_wilson=list(wilson(sum(yes), len(yes))) if yes else None,
            sec_per_image=float(np.median([r["sec"] for r in rows])))
    # paired CLIP differences against g = 7.5
    base = {(r["p"], r["seed"]): r["clip"] for r in G if r["g"] == 7.5}
    R["guidance_diff_vs_7.5"] = {}
    for g in gs:
        d = {}
        for r in G:
            if r["g"] == g and (r["p"], r["seed"]) in base:
                d.setdefault(r["p"], []).append(r["clip"] - base[(r["p"], r["seed"])])
        if d:
            R["guidance_diff_vs_7.5"][str(g)] = boot_prompts([np.mean(v) for v in d.values()])
    # saturation proxy: mean HSV saturation and pixel std per image
    R["guidance_color"] = {}
    for g in gs:
        sat = []
        for r in G:
            if r["g"] == g and not r["nsfw"]:
                im = Image.open(f"runs/img/guidance/{r['key']}.png").convert("HSV")
                sat.append(np.asarray(im)[..., 1].mean() / 255)
        R["guidance_color"][str(g)] = [float(np.mean(sat)), float(np.std(sat))]

# ---------------- judge validation against hand labels
if hand and judge:
    agree, rows = [], []
    for k, h in hand.items():
        if k in judge and judge[k].get("verdict"):
            agree.append((judge[k]["verdict"] == "yes") == h)
            rows.append(((judge[k]["verdict"] == "yes"), h))
    if rows:
        a = np.array(rows)
        po = np.mean(a[:, 0] == a[:, 1]); pj, ph = a[:, 0].mean(), a[:, 1].mean()
        pe = pj * ph + (1 - pj) * (1 - ph)
        R["judge_validation"] = dict(n=len(rows), agree=list(wilson(sum(agree), len(agree))),
                                     kappa=float((po - pe) / (1 - pe)) if pe < 1 else None,
                                     judge_yes=float(pj), hand_yes=float(ph),
                                     tp=int(((a[:, 0] == 1) & (a[:, 1] == 1)).sum()),
                                     fp=int(((a[:, 0] == 1) & (a[:, 1] == 0)).sum()),
                                     fn=int(((a[:, 0] == 0) & (a[:, 1] == 1)).sum()),
                                     tn=int(((a[:, 0] == 0) & (a[:, 1] == 0)).sum()))
    # hand-label yes rate per guidance scale
    R["hand_by_g"] = {}
    gk = {r["key"]: r["g"] for r in G}
    for g in sorted({gk[k] for k in hand if k in gk}):
        v = [hand[k] for k in hand if gk.get(k) == g]
        R["hand_by_g"][str(g)] = list(wilson(sum(v), len(v))) + [len(v)]
if judge:
    stops = [j.get("stop") for j in judge.values()]
    R["judge_meta"] = dict(n=len(judge), unparsed=sum(1 for j in judge.values() if not j.get("verdict")),
                           max_tokens_stops=sum(1 for s in stops if s == "max_tokens"),
                           errors=sum(1 for j in judge.values() if j.get("error")))

# ---------------- steps sweep: PSNR and CLIP-embedding similarity to a 100-step DPM-Solver++ reference
S = load("steps")
if S:
    ref = {r["p"]: r for r in S if r["sampler"] == "dpmpp2m" and r["steps"] == 100}
    def psnr(a, b):
        x = np.asarray(Image.open(a), np.float32); y = np.asarray(Image.open(b), np.float32)
        mse = ((x - y) ** 2).mean()
        return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 99.0
    R["steps"] = {}
    for smp in ["ddim", "euler", "dpmpp2m"]:
        for st in sorted({r["steps"] for r in S if r["sampler"] == smp and r["steps"] != 100}):
            rows = [r for r in S if r["sampler"] == smp and r["steps"] == st and r["p"] in ref and not r["nsfw"]]
            if not rows:
                continue
            ps = [psnr(f"runs/img/steps/{r['key']}.png", f"runs/img/steps/{ref[r['p']]['key']}.png") for r in rows]
            cs = [float(emb("steps", r["key"]) @ emb("steps", ref[r["p"]]["key"]) /
                        (np.linalg.norm(emb("steps", r["key"])) * np.linalg.norm(emb("steps", ref[r["p"]]["key"])))) for r in rows]
            R["steps"][f"{smp}_{st}"] = dict(sampler=smp, steps=st, n=len(rows), psnr=boot_prompts(ps),
                                             clip_cos_to_ref=boot_prompts(cs), clip=boot_prompts([r["clip"] for r in rows]),
                                             sec_per_image=float(np.median([r["sec"] for r in rows])))
    R["steps_ref_clip"] = boot_prompts([r["clip"] for r in ref.values()])
    R["steps_ref_n"] = len(ref)

# ---------------- negative prompts
N = load("negative")
if N:
    pairs = {}
    for r in N:
        pairs.setdefault((r["p"], r["seed"]), {})["neg" if r["neg"] else "plain"] = r
    d_concept, d_clip, by_p = [], [], {}
    for (p, s), v in pairs.items():
        if "neg" in v and "plain" in v and not v["neg"]["nsfw"] and not v["plain"]["nsfw"]:
            by_p.setdefault(p, []).append((v["neg"]["concept_sim"] - v["plain"]["concept_sim"],
                                           v["neg"]["clip"] - v["plain"]["clip"]))
    R["negative"] = dict(
        pairs=sum(len(v) for v in by_p.values()), prompts=len(by_p),
        concept_sim_plain=float(np.mean([r["concept_sim"] for r in N if not r["neg"]])),
        concept_sim_neg=float(np.mean([r["concept_sim"] for r in N if r["neg"]])),
        concept_diff=boot_prompts([np.mean([x[0] for x in v]) for v in by_p.values()]),
        clip_diff=boot_prompts([np.mean([x[1] for x in v]) for v in by_p.values()]),
        per_prompt={str(p): float(np.mean([x[0] for x in v])) for p, v in by_p.items()},
        nsfw=sum(r["nsfw"] for r in N))

# ---------------- safety checker over every image generated
allrecs = G + S + N
if allrecs:
    m = np.array([r["safety_margin"] for r in allrecs])
    R["safety"] = dict(images=len(allrecs), flagged=int(sum(r["nsfw"] for r in allrecs)),
                       margin_max=float(m.max()), margin_median=float(np.median(m)),
                       margin_p99=float(np.percentile(m, 99)))
json.dump(R, open("runs/results.json", "w"), indent=1)
print(json.dumps(R, indent=1)[:6000])
