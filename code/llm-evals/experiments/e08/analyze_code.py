"""pass@k on HumanEval vs HumanEval+ from the graded runs; writes runs/code_results.json."""
import json, sys, ast, re, collections
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify, stats
R = "/home/amit/.cache/vidyalai-scratch/e08/runs/"; D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
he = json.load(open(D + "e08-humaneval.json")); hp = json.load(open(D + "e08-humanevalplus.json"))
# tests per problem: asserts in the original check(), inputs in the HumanEval+ check()
base_n = [len(re.findall(r"^\s*assert", t["test"], re.M)) for t in he]
plus_n = [len(ast.literal_eval(t["test"].split("inputs = ")[1].split("\n")[0])) for t in hp]
out = dict(tests=dict(base_mean=float(np.mean(base_n)), base_median=float(np.median(base_n)),
                      plus_mean=float(np.mean(plus_n)), plus_median=float(np.median(plus_n)),
                      plus_min=int(min(plus_n)), plus_max=int(max(plus_n))))
SKIP = 32  # HumanEval/32: the HF HumanEval+ check is broken; its plus result is set to the base result
def load(name):
    g = json.load(open(R + f"he_{name}.graded.json"))
    for r in g:
        if r["task"] == SKIP:
            r["plus"] = r["base"]
    return g
for name in ("haiku", "sonnet", "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
    g = load(name)
    n = len(g) // 164
    B = np.zeros((164, n)); P = np.zeros((164, n))
    for r in g:
        s = int(r["id"].split("-s")[1]); B[r["task"], s] = r["base"] == "pass"; P[r["task"], s] = r["plus"] == "pass"
    res = dict(n=n, stops=collections.Counter(r["stop"] for r in g),
               plus_fail_kinds=collections.Counter(r["plus"] + " " + (r["plus_err"] or "").split(":")[0] for r in g if r["base"] == "pass" and r["plus"] != "pass"),
               base_fail_kinds=collections.Counter(r["base"] + " " + (r["base_err"] or "").split(":")[0] for r in g if r["base"] != "pass"))
    ks = [k for k in (1, 2, 5, 10) if k <= n]
    res["base"] = verify.pass_at_k_table(B.sum(1), n, ks); res["plus"] = verify.pass_at_k_table(P.sum(1), n, ks)
    # paired drop in pass@1: per-problem mean over samples, base minus plus
    a, b = B.mean(1), P.mean(1)
    pdf = stats.paired_diff(b, a)
    res["drop_pass1"] = pdf
    res["drop_boot"] = stats.paired_bootstrap(b, a)
    for k in ks:
        pa = np.array([verify.pass_at_k(n, int(c), k) for c in B.sum(1)]); pb = np.array([verify.pass_at_k(n, int(c), k) for c in P.sum(1)])
        res[f"drop_boot_k{k}"] = (float((pa - pb).mean()), stats.paired_bootstrap(pb, pa))
    res["problems_any_base_pass_plus_fail"] = int(((B - P) > 0).any(1).sum())
    res["samples_base_pass_plus_fail"] = int(((B - P) > 0).sum())
    res["problems_all_base_some_plus_fail"] = int(((B.sum(1) == n) & (P.sum(1) < n)).sum())
    res["problems_all_base_all_plus_fail"] = int(((B.sum(1) == n) & (P.sum(1) == 0)).sum())
    res["per_problem_base"] = B.sum(1).astype(int).tolist(); res["per_problem_plus"] = P.sum(1).astype(int).tolist()
    if n > 1:
        res["var_split_base"] = stats.variance_split(B); res["var_split_plus"] = stats.variance_split(P)
        res["flaky_base"] = int(((B.sum(1) > 0) & (B.sum(1) < n)).sum()); res["always_base"] = int((B.sum(1) == n).sum()); res["never_base"] = int((B.sum(1) == 0).sum())
        # pass@1 from just the first sample, vs all n: spread of single-sample benchmark runs
        runs = B.mean(0); res["single_run_scores_base"] = runs.tolist(); runsp = P.mean(0); res["single_run_scores_plus"] = runsp.tolist()
    else:
        res["wilson_base"] = stats.wilson(int(B.sum()), 164); res["wilson_plus"] = stats.wilson(int(P.sum()), 164)
        res["mcnemar"] = stats.mcnemar(B[:, 0], P[:, 0])
    out[name] = res
json.dump(out, open(R + "code_results.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
for k, v in out.items():
    if k == "tests":
        print(v); continue
    print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("per_problem") and not kk.startswith("single")})
