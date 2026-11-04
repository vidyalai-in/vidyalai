"""Summaries for the article: caching experiment (runs/cache_*.jsonl) -> runs/results_cache.json."""
import json, re, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import serving as S, ledger

D = "/home/amit/.cache/vidyalai-scratch/v13"
P = ledger.PRICES["claude-haiku-5-5"]
res = {}
for size in ("S", "L", "XL"):
    try:
        recs = [json.loads(l) for l in open(f"{D}/runs/cache_{size}.jsonl")]
    except FileNotFoundError:
        continue
    recs = [r for r in recs if r["round"] >= 0]
    out = {}
    for arm in ("plain", "write", "hit"):
        rs = [r for r in recs if r["arm"] == arm and "error" not in r]
        errs = [r for r in recs if r["arm"] == arm and "error" in r]
        tt = [r["ttft"] for r in rs if r["ttft"] is not None]
        tot = [r["total"] for r in rs]
        u = lambda k: [r["usage"].get(k) or 0 for r in rs]
        last = lambda s: (re.findall(r"-?\d+", s.replace(",", "")) or [None])[-1]
        ok = sum(last(r["text"]) == r["answer"] for r in rs)
        worked = sum(len(r["text"].strip()) > len(r["answer"]) + 1 for r in rs)
        out[arm] = {
            "n": len(rs), "errors": len(errs),
            "ttft_median": S.median_ci(tt), "ttft_p90": S.quantile_ci(tt, 0.9),
            "total_median": S.median_ci(tot),
            "input_tokens": float(np.median(u("input_tokens"))),
            "cache_write_tokens": float(np.median(u("cache_creation_input_tokens"))),
            "cache_read_tokens": float(np.median(u("cache_read_input_tokens"))),
            "output_tokens_median": float(np.median(u("output_tokens"))),
            "output_tokens_max": int(max(u("output_tokens"))),
            "cut_off": sum(r["stop"] == "max_tokens" for r in rs),
            "cost_mean": float(np.mean([r["cost"] for r in rs])),
            "cost_total": float(np.sum([r["cost"] for r in rs])),
            "correct": ok, "acc": S.wilson(ok, len(rs)), "worked_out_loud": worked,
        }
    out["hit_minus_plain_ttft"] = S.diff_median_ci([r["ttft"] for r in recs if r["arm"] == "plain" and r.get("ttft")],
                                                   [r["ttft"] for r in recs if r["arm"] == "hit" and r.get("ttft")])
    out["write_minus_plain_ttft"] = S.diff_median_ci([r["ttft"] for r in recs if r["arm"] == "plain" and r.get("ttft")],
                                                     [r["ttft"] for r in recs if r["arm"] == "write" and r.get("ttft")])
    # paired by round: hit minus plain, write minus plain (same question, adjacent calls)
    byr = {}
    for r in recs:
        if "error" not in r and r.get("ttft"):
            byr.setdefault(r["round"], {})[r["arm"]] = r["ttft"]
    rng = np.random.default_rng(0)
    for other in ("hit", "write"):
        d = np.array([v[other] - v["plain"] for v in byr.values() if other in v and "plain" in v])
        b = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
        out[f"paired_{other}_minus_plain"] = {"median": float(np.median(d)), "ci": np.percentile(b, [2.5, 97.5]).tolist(),
                                              "faster": int((d < 0).sum()), "n": int(len(d))}
    # break-even from list prices and the measured prefix size
    T = out["hit"]["cache_read_tokens"]
    q_in = out["hit"]["input_tokens"]
    o = out["plain"]["output_tokens_median"]
    plain_q = (T + q_in) * P["in"] + o * P["out"]
    first = T * P["cache_write"] + q_in * P["in"] + o * P["out"]
    later = T * P["cache_read"] + q_in * P["in"] + o * P["out"]
    curve = []
    for n in range(1, 11):
        curve.append({"n": n, "plain": n * plain_q / 1e6, "cached": (first + (n - 1) * later) / 1e6})
    out["breakeven"] = {"prefix_tokens": T, "per_q_plain_usd": plain_q / 1e6, "first_cached_usd": first / 1e6,
                        "later_cached_usd": later / 1e6, "curve": curve,
                        "first_n_cheaper": next(c["n"] for c in curve if c["cached"] < c["plain"])}
    # same-question pairs: did the answer change between arms?
    by = {}
    for r in recs:
        if "error" not in r:
            by.setdefault(r["round"], {})[r["arm"]] = r["text"].strip()
    out["same_answer_all_arms"] = sum(len(set(v.values())) == 1 for v in by.values() if len(v) == 3)
    out["rounds_complete"] = sum(len(v) == 3 for v in by.values())
    walls = [r["wall"] for r in recs]
    out["wall_span_min"] = (max(walls) - min(walls)) / 60
    out["first_wall"] = min(walls)
    res[size] = out
json.dump(res, open(f"{D}/runs/results_cache.json", "w"), indent=1)
for s, o in res.items():
    print(s, {a: (o[a]["n"], o[a]["errors"], [round(x, 3) for x in o[a]["ttft_median"]], [round(x, 3) for x in o[a]["ttft_p90"]],
                  o[a]["input_tokens"], o[a]["cache_write_tokens"], o[a]["cache_read_tokens"], round(o[a]["cost_mean"] * 1e6, 1),
                  o[a]["correct"], o[a]["cut_off"]) for a in ("plain", "write", "hit")})
    print("  hit-plain", [round(x, 3) for x in o["hit_minus_plain_ttft"]], "write-plain", [round(x, 3) for x in o["write_minus_plain_ttft"]],
          "breakeven n", o["breakeven"]["first_n_cheaper"], "same answers", o["same_answer_all_arms"], "/", o["rounds_complete"],
          "span min", round(o["wall_span_min"], 1), "paired", o["paired_hit_minus_plain"], o["paired_write_minus_plain"])
