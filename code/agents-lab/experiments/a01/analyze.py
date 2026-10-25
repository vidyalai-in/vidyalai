"""Summaries for Part 1. Usage: python analyze.py [arms...]"""
import json, math, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

def load(arm):
    p = HERE / "runs" / f"{arm}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def boot_tasks(runs, B=4000, seed=0):
    """95% CI for accuracy, resampling tasks (each with all its repeats)."""
    by = defaultdict(list)
    for r in runs: by[r["task"]].append(r["correct"])
    ids = list(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        s = [rng.choice(ids) for _ in ids]
        vals.append(sum(sum(by[t]) for t in s) / sum(len(by[t]) for t in s))
    vals.sort(); return vals[int(.025*B)], vals[int(.975*B)]

def summary(arm, runs, label=None):
    n = len(runs); k = sum(r["correct"] for r in runs)
    lo, hi = wilson(k, n); blo, bhi = boot_tasks(runs)
    steps = [len(r["steps"]) for r in runs]
    cost = [r["cost"] for r in runs]
    lat = [s["latency_s"] for r in runs for s in r["steps"]]
    wall = [sum(s["latency_s"] for s in r["steps"]) for r in runs]
    ctx = [s["input_tokens"] + s["cache_read"] + s["cache_write"] for r in runs for s in r["steps"]]
    tot_in = [sum(s["input_tokens"] + s["cache_read"] + s["cache_write"] for s in r["steps"]) for r in runs]
    out = [s["output_tokens"] for r in runs for s in r["steps"]]
    print(f"== {label or arm}: n={n} correct={k} acc={k/n:.3f} wilson=[{lo:.3f},{hi:.3f}] taskboot=[{blo:.3f},{bhi:.3f}]")
    print(f"   steps/task mean={st.mean(steps):.2f} median={st.median(steps)} max={max(steps)}  stops={dict((s, sum(r['stop']==s for r in runs)) for s in set(r['stop'] for r in runs))}")
    print(f"   cost/task mean=${st.mean(cost):.6f} total=${sum(cost):.4f}  latency/step mean={st.mean(lat):.2f}s median={st.median(lat):.2f}s  model-time/task mean={st.mean(wall):.2f}s")
    print(f"   context/step mean={st.mean(ctx):.0f}  total input/task mean={st.mean(tot_in):.0f}  output/step mean={st.mean(out):.0f} median={st.median(out)}")
    byk = defaultdict(list)
    for r in runs: byk[r["kind"]].append(r)
    for kind, rs in sorted(byk.items()):
        kk = sum(r["correct"] for r in rs); l, h = wilson(kk, len(rs))
        print(f"   {kind:9s} {kk}/{len(rs)} = {kk/len(rs):.3f} [{l:.3f},{h:.3f}]  steps={st.mean(len(r['steps']) for r in rs):.2f}  cost=${st.mean(r['cost'] for r in rs):.6f}")
    return runs

def per_step(runs):
    by = defaultdict(list)
    for r in runs:
        for i, s in enumerate(r["steps"], 1):
            by[i].append(s)
    print("   step  n   context  cache_read  cache_write  uncached  out   latency")
    for i in sorted(by):
        ss = by[i]
        print(f"   {i:4d} {len(ss):3d} {st.mean(x['input_tokens']+x['cache_read']+x['cache_write'] for x in ss):8.0f} "
              f"{st.mean(x['cache_read'] for x in ss):10.0f} {st.mean(x['cache_write'] for x in ss):11.0f} "
              f"{st.mean(x['input_tokens'] for x in ss):8.0f} {st.mean(x['output_tokens'] for x in ss):5.0f} {st.mean(x['latency_s'] for x in ss):6.2f}s")

def per_task(runs):
    by = defaultdict(list)
    for r in runs: by[r["task"]].append(r)
    for t in sorted(by):
        rs = by[t]
        print(f"   {t} {rs[0]['kind'][:5]} {sum(r['correct'] for r in rs)}/{len(rs)} steps={[len(r['steps']) for r in rs]} "
              f"answers={[ (r['answer'].strip().splitlines() or [''])[-1][:40] for r in rs if not r['correct']][:2]}")

if __name__ == "__main__":
    arms = sys.argv[1:] or ["agent", "agent_nocache", "rag", "stuff", "sonnet"]
    for a in arms:
        runs = load(a)
        if not runs: continue
        summary(a, runs)
        if a in ("agent", "agent_nocache", "sonnet"):
            per_step(runs)
        per_task(runs)
