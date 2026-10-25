import json, statistics as st, collections
from common import ci95, ledger
R = [json.loads(l) for l in open("runs/exp3_strict.jsonl")]
for strict in (False, True):
    A = [d for d in R if d["strict"] == strict]
    calls = [c for d in A for c in d["calls"]]
    bad = [c for c in calls if c["schema_errors"]]
    bad_other = [c for c in bad if not all("assignee_email" in e for e in c["schema_errors"])]
    rng = [c for c in calls if c["range_errors"]]
    nocall = [d for d in A if d["n_calls"] == 0]
    multi = [d for d in A if d["n_calls"] > 1]
    wrong = [d for d in A if d["calls"] and d["calls"][0]["name"] != d["expected"]]
    secs = [d["secs"] for d in A]
    cost = sum(ledger.cost_of("claude-haiku-5-5", d["usage"]) for d in A)
    print(f"strict={strict}: responses {len(A)}, calls {len(calls)}, no-call {len(nocall)}, multi-call {len(multi)}, wrong tool {len(wrong)}")
    lo, hi = ci95(len(bad), len(calls))
    print(f"  schema-invalid calls {len(bad)}/{len(calls)} = {len(bad)/len(calls):.1%} CI [{lo:.1%}, {hi:.1%}]")
    lo, hi = ci95(len(bad_other), len(calls))
    print(f"  excluding nullable-email field: {len(bad_other)}/{len(calls)} CI [{lo:.1%}, {hi:.1%}]")
    lo, hi = ci95(len(rng), len(calls))
    print(f"  range violations {len(rng)}/{len(calls)} CI [{lo:.1%},{hi:.1%}]")
    print("  error kinds:", collections.Counter(e.split(":")[0] + ":" + e.split(":")[1][:40] for c in bad for e in c["schema_errors"]).most_common(12))
    print("  range kinds:", collections.Counter(e for c in rng for e in c["range_errors"]).most_common(10))
    print(f"  latency mean {st.mean(secs):.2f}s median {st.median(secs):.2f}s p90 {sorted(secs)[int(.9*len(secs))]:.2f}")
    print(f"  input tok mean {st.mean(d['usage']['input_tokens'] for d in A):.0f} output tok mean {st.mean(d['usage']['output_tokens'] for d in A):.0f} (sd {st.stdev(d['usage']['output_tokens'] for d in A):.0f})")
    print(f"  thinking present {sum(d['has_thinking'] for d in A)}/{len(A)}; cost ${cost:.4f}  per call ${cost/len(A):.5f}")
    print("  nocall idx", collections.Counter(d['idx'] for d in nocall), "multi idx", collections.Counter(d['idx'] for d in multi))
    print("  wrong", [(d['idx'], d['calls'][0]['name']) for d in wrong][:10])
    print("  bad idx", collections.Counter(next(i for i in [0]) or 0 for c in bad))
