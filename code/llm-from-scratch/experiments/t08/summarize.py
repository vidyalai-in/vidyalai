"""summarize.py <run dir>...: steps, tokens/s (median of steps >= 3), peak memory, last val."""
import json, statistics, sys
for d in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    start = next(r for r in recs if r.get("event") == "start")
    steps = [r for r in recs if "loss" in r]
    vals = [r for r in recs if "val_loss" in r]
    end = next((r for r in recs if r.get("event") == "end"), {})
    tps = [r["tok_per_s"] for r in steps if r["step"] >= 3]
    a = start["args"]
    print(f"{d.rstrip('/').split('/')[-1]:28} {a['model']:5} mb={a['micro_batch']:<3} tok/step={start['tokens_per_step']:<7} "
          f"lr={a['lr']:<7} steps={len(steps):<5} tok/s(med)={statistics.median(tps) if tps else 0:<7.0f} "
          f"peak={end.get('peak_mem_gib')} GiB  val={vals[-1]['val_loss'] if vals else None}  "
          f"train(last10)={statistics.mean(r['loss'] for r in steps[-10:]):.4f} elapsed={end.get('elapsed')}")
