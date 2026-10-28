# Run facts from a DPO log.jsonl and its nvidia-smi log: steps, pairs, tokens, wall time, throughput,
# clocks and power, final held-out metrics. usage: python run_facts.py <run name> [smi tag]
import sys, json, statistics
R = '/home/amit/.cache/vidyalai-scratch/t11/runs/'
L = '/home/amit/.cache/vidyalai-scratch/t11/logs/'
name = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else name
recs = [json.loads(l) for l in open(R + name + '/log.jsonl')]
steps = [r for r in recs if 'loss' in r]
vals = [r for r in recs if 'val' in r]
end = [r for r in recs if r.get('event') == 'end'][-1]
start = [r for r in recs if r.get('event') in ('start', 'resume')]
real_dt = sum(r['dt'] for r in steps)
out = {"steps": len(steps), "pairs": sum(r['pairs'] for r in steps), "padded": sum(r['padded'] for r in steps),
       "wall_s": end['elapsed'], "train_s": round(real_dt, 1), "padded_tok_s": round(sum(r['padded'] for r in steps) / real_dt),
       "pairs_per_step_mean": statistics.mean(r['pairs'] for r in steps),
       "clipped": sum(r['grad_norm'] > 1.0 for r in steps), "grad_norm_median": statistics.median(r['grad_norm'] for r in steps),
       "peak_mem_gib": end['peak_mem_gib'], "resumed": len(start) > 1,
       "val_first": vals[0]['val'], "val_last": vals[-1]['val'],
       "train_last10": {k: statistics.mean(r[k] for r in steps[-10:]) for k in ('loss', 'acc', 'margin', 'chosen_reward', 'rejected_reward')}}
try:
    rows = [l.strip().split(', ') for l in open(L + f'smi_{tag}.csv') if l.strip()]
    sm = sorted(float(r[1]) for r in rows); mem = [r[2] for r in rows]; pw = [float(r[3]) for r in rows]
    tmp = sorted(float(r[5]) for r in rows)
    n = len(sm)
    out["smi"] = {"samples": n, "sm_median": sm[n // 2], "sm_p5": sm[n // 20], "sm_p95": sm[n * 19 // 20],
                  "mem_clocks": sorted(set(mem)), "mem_7000_share": mem.count('7000') / n, "power_mean": statistics.mean(pw),
                  "power_limit": sorted(set(r[4] for r in rows)), "temp_median": tmp[n // 2]}
except FileNotFoundError:
    pass
print(json.dumps(out, indent=1))
