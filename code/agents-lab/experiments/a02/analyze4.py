import json, statistics as st, collections, math
from common import ci95, ledger
R = [json.loads(l) for l in open("runs/exp4_parallel.jsonl")]
def mci(xs):
    m = st.mean(xs); s = st.stdev(xs) if len(xs) > 1 else 0
    return m, 1.96 * s / math.sqrt(len(xs))
for kind in ("indep", "dep", "mixed", "all"):
    for dis in (False, True):
        A = [d for d in R if d["disable"] == dis and (kind == "all" or d["kind"] == kind)]
        cost = sum(ledger.cost_of("claude-haiku-5-5", u) for d in A for u in d["usage"])
        intok = [sum(u["input_tokens"] for u in d["usage"]) for d in A]
        outtok = [sum(u["output_tokens"] for u in d["usage"]) for d in A]
        batched = sum(1 for d in A if d["tool_turns"] == d["min_tool_turns"])
        lo, hi = ci95(batched, len(A))
        multi_turns = sum(1 for d in A for c in d["calls_per_turn"] if c > 1); all_turns = sum(len(d["calls_per_turn"]) for d in A)
        print(f"{kind:5s} {'seq' if dis else 'par'} n={len(A)} api_calls {mci([d['api_calls'] for d in A])[0]:.2f}±{mci([d['api_calls'] for d in A])[1]:.2f} "
              f"tool_calls {st.mean(d['tool_calls'] for d in A):.2f} secs {mci([d['secs'] for d in A])[0]:.2f}±{mci([d['secs'] for d in A])[1]:.2f} "
              f"in_tok {st.mean(intok):.0f} out_tok {st.mean(outtok):.0f} cost/task ${cost/len(A):.5f} total ${cost:.4f} "
              f"min-turns {batched}/{len(A)} [{lo:.0%},{hi:.0%}] multi-call turns {multi_turns}/{all_turns} tools/turn {sum(d['tool_calls'] for d in A)/all_turns:.2f}")
# dependent tasks in parallel mode: did any call guess an argument before seeing the order?
for d in R:
    if not d["disable"] and d["kind"] != "indep" and d["tool_turns"] != d["min_tool_turns"]:
        print("off-min", d["idx"], d["calls_per_turn"], [[c["name"] for c in t] for t in d["per_turn"]])
