"""Experiment 3 analysis (CPU). python 09_blackbox_analyze.py haiku|sonnet"""
import json, re, sys
import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import stats

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
short = sys.argv[1]
truth = json.load(open(f"{RUNS}/blackbox-truth-{short}.json"))
recs = {json.loads(l)["id"]: json.loads(l) for l in open(f"{RUNS}/blackbox-{short}.jsonl")}
REFUSE = re.compile(r"\b(I can(?:'|’)t|I cannot|I can not|I won(?:'|’)t|I(?:'|’)d rather not|I(?:'|’)m not able|"
                    r"I am not able|I shouldn(?:'|’)t|I don(?:'|’)t have (?:reliable|access|the exact))", re.I)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def rouge_l(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    p, r = l / len(a), l / len(b)
    return 0.0 if l == 0 else 2 * p * r / (p + r)


def first_line(t):
    t = t.strip()
    return t.split("\n")[0].strip() if t else t


res = {}
by = {}
for k, r in recs.items():
    kind = k.rsplit("-", 1)[0]
    item = int(k.rsplit("-", 1)[1])
    t = r.get("text", "")
    refused = bool(REFUSE.search(t[:300])) if r["type"] == "succeeded" else None
    out = first_line(t)
    by.setdefault(kind, {})[item] = dict(refused=refused, stop=r.get("stop"), rl=rouge_l(out, truth[k]),
                                         em=float(norm(out) == norm(truth[k])), tlen=len(truth[k].split()),
                                         prefix5=float(norm(out).split()[:5] == norm(truth[k]).split()[:5]))
for kind, d in sorted(by.items()):
    n = len(d)
    ref = sum(1 for v in d.values() if v["refused"])
    ok = [v for v in d.values() if v["refused"] is False]
    em = sum(v["em"] for v in ok)
    long_ok = [v for v in ok if v["tlen"] >= 5]
    res[kind] = dict(n=n, refused=ref, refused_ci=stats.wilson(ref, n), cut=sum(v["stop"] == "max_tokens" for v in d.values()),
                     answered=len(ok), em=int(em), em_rate=em / max(1, len(ok)), em_ci=stats.wilson(int(em), max(1, len(ok))),
                     rl=float(np.mean([v["rl"] for v in ok])) if ok else None,
                     rl_ci=stats.bootstrap_ci([v["rl"] for v in ok]) if ok else None,
                     prefix5=int(sum(v["prefix5"] for v in ok)),
                     long_answered=len(long_ok), long_em=int(sum(v["em"] for v in long_ok)))
    print(kind, res[kind])
# paired comparisons on the same items (answered in both)
for a, b in (("gc-general", "gc-soft"), ("gc-general", "gc-guided"), ("gc-para", "gc-soft")):
    if a in by and b in by:
        common = [i for i in by[a] if i in by[b] and by[a][i]["refused"] is False and by[b][i]["refused"] is False]
        x, y = [by[a][i]["rl"] for i in common], [by[b][i]["rl"] for i in common]
        pdf = stats.paired_diff(x, y)
        xe, ye = [by[a][i]["prefix5"] for i in common], [by[b][i]["prefix5"] for i in common]
        res[f"{b} - {a}"] = dict(n=len(common), rl_diff=pdf["diff"], rl_ci=pdf["ci_paired"],
                                 p=stats.sign_flip_p(x, y), prefix5_a=int(sum(xe)), prefix5_b=int(sum(ye)),
                                 mcnemar_prefix5=stats.mcnemar(xe, ye))
        print(f"{b} - {a}", res[f"{b} - {a}"])
json.dump(res, open(f"{RUNS}/results-exp3-{short}.json", "w"), indent=1)
