import json, statistics as st, collections
from common import ci95, ledger
R = [json.loads(l) for l in open("runs/exp5_descriptions.jsonl")]
for det in (False, True):
    A = [d for d in R if d["detailed"] == det]
    ok = sum(d["first_ok"] for d in A); lo, hi = ci95(ok, len(A))
    nocall = sum(1 for d in A if not d["called"]); multi = sum(1 for d in A if len(d["called"]) > 1)
    cost = sum(ledger.cost_of("claude-haiku-5-5", d["usage"]) for d in A)
    print(f"detailed={det} acc {ok}/{len(A)} = {ok/len(A):.1%} [{lo:.1%},{hi:.1%}] nocall {nocall} multi {multi} in_tok {st.mean(d['usage']['input_tokens'] for d in A):.0f} out {st.mean(d['usage']['output_tokens'] for d in A):.0f} secs {st.mean(d['secs'] for d in A):.2f} cost ${cost:.4f}")
    perq = collections.defaultdict(list)
    for d in A: perq[d["idx"]].append(d["first_ok"])
    print("  tasks always right", sum(all(v) for v in perq.values()), "/30; tasks wrong at least once", [k for k, v in perq.items() if not all(v)])
