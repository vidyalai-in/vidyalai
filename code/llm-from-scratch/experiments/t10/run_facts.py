import json, statistics as st, sys, csv
for tag in sys.argv[1:]:
    L = [json.loads(l) for l in open(f'runs/{tag}/log.jsonl')]
    s = [r for r in L if 'loss' in r]
    end = [r for r in L if r.get('event') == 'end'][0]
    f = {"steps": len(s), "padded": sum(r['padded'] for r in s), "real": sum(r['tokens'] for r in s),
         "targets": sum(r['targets'] for r in s), "step_time": sum(r['dt'] for r in s),
         "median_tok_s": st.median(r['tok_per_s'] for r in s), "elapsed": end['elapsed'], "peak": end['peak_mem_gib'],
         "max_gn": max((r['grad_norm'], r['step']) for r in s), "clipped": sum(r['grad_norm'] > 1 for r in s),
         "first_loss": s[0]['loss'], "last10_loss": st.mean(r['loss'] for r in s[-10:]),
         "targets_per_step_range": (min(r['targets'] for r in s), max(r['targets'] for r in s))}
    f["real_tok_s"] = f["real"] / f["step_time"]; f["padded_tok_s"] = f["padded"] / f["step_time"]
    rows = [r for r in csv.reader(open(f'logs/smi_{tag}.csv'))]
    sm = sorted(float(r[1]) for r in rows if r[1].strip().isdigit()); mem = [r[2].strip() for r in rows]
    pw = [float(r[3]) for r in rows if r[3].strip().replace('.', '').isdigit()]
    f["smi"] = {"n": len(rows), "sm_median": st.median(sm), "sm_p5": sm[len(sm)//20], "sm_p95": sm[-len(sm)//20],
                "mem_clocks": {m: mem.count(m) for m in set(mem)}, "power_mean": st.mean(pw),
                "power_limit": sorted(set(r[4].strip() for r in rows)), "temp_median": st.median(float(r[5]) for r in rows)}
    print(tag, json.dumps(f, indent=0))
