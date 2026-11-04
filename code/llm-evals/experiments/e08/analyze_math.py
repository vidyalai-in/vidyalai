"""MATH-500: accuracy under three graders per setting, paired grader gaps, hand-check agreement."""
import json, sys, collections
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import stats
R = "/home/amit/.cache/vidyalai-scratch/e08/runs/"
rows = json.load(open(R + "math_graded.json"))
# Hand check (read by the writer): every reply where the graders disagree was read, and every reply all three
# rejected. Disagreements: symbolic was right every time. All-reject replies that were in fact correct:
HAND_RIGHT = {(100, "145°"), (103, "3 < \\lambda \\le 4"), (103, "3<\\lambda\\le 4"), (248, "-\\mathbf{I}"),
              (379, "12^{\\mathrm{th}}\\text{ grade}"), (392, "√66"), (409, "2√5"), (415, "9 \\pmod{11}"),
              (428, "Minimum value: 64")}
# Effort high: 47 replies rejected by all graders were read in full; in 37 the extractor had picked the wrong span
# (a check line, a step heading) and the reply's stated answer was right. 7 were wrong. (3 more are in HAND_RIGHT.)
HIGH_RIGHT = {61, 379, 394, 339, 120, 82, 50, 361, 60, 133, 410, 99, 244, 302, 396, 454, 483, 25, 251, 10,
              349, 364, 315, 186, 91, 297, 338, 64, 308, 71, 96, 287, 403, 324, 422, 240, 190}
for r in rows:
    disagree = len({r["exact"], r["normalized"], r["symbolic"]}) > 1
    r["hand"] = True if disagree else (r["symbolic"] or (r["item"], r["pred"]) in HAND_RIGHT
                                       or (r["setting"] == "high" and r["item"] in HIGH_RIGHT))
    r["checked"] = disagree or not r["symbolic"]
out = {}
G = ("exact", "normalized", "symbolic", "hand", "exact_boxonly", "normalized_boxonly", "symbolic_boxonly", "symbolic_full")
for key in sorted({(r["setting"], r["rep"]) for r in rows}):
    rr = sorted([r for r in rows if (r["setting"], r["rep"]) == key], key=lambda r: r["item"])
    subj = [r["subject"] for r in rr]
    res = dict(n=len(rr), rules=collections.Counter(r["rule"] for r in rr), trunc=sum(r["stop"] == "max_tokens" for r in rr),
               thinking=sum(r["thinking"] for r in rr), out_tokens_mean=float(np.mean([r["out_tokens"] for r in rr])),
               out_tokens_median=float(np.median([r["out_tokens"] for r in rr])))
    for g in G:
        s = np.array([r[g] for r in rr], float)
        res[g] = (float(s.mean()), *stats.wilson(int(s.sum()), len(s)))
    for a, b in (("exact", "normalized"), ("normalized", "symbolic"), ("exact", "symbolic"), ("symbolic", "hand"), ("symbolic_boxonly", "symbolic")):
        x = np.array([r[a] for r in rr], float); y = np.array([r[b] for r in rr], float)
        res[f"{b}-{a}"] = (float((y - x).mean()), stats.paired_bootstrap(x, y))
    out["-".join(key)] = res
# settings compared under each grader (paired by item, repeat 0)
def vec(setting, rep, g):
    return np.array([r[g] for r in sorted([r for r in rows if r["setting"] == setting and r["rep"] == rep], key=lambda r: r["item"])], float)
cmp = {}
for g in ("exact", "normalized", "symbolic", "hand"):
    for s in ("low", "high"):
        if not any(r["setting"] == s for r in rows): continue
        a, b = vec("off", "r0", g), vec(s, "r0", g)
        cmp[f"{s}-off {g}"] = (float((b - a).mean()), stats.paired_bootstrap(a, b), stats.mcnemar(a, b))
    a, b = vec("off", "r0", g), vec("off", "r1", g)
    cmp[f"off r1-r0 {g}"] = (float((b - a).mean()), stats.paired_bootstrap(a, b))
out["compare"] = cmp
# hand-check agreement on the checked replies
chk = [r for r in rows if r["checked"]]
out["hand_check"] = dict(n=len(chk), unique=len({(r["item"], r["pred"]) for r in chk}),
                         **{g: sum(r[g] == r["hand"] for r in chk) for g in ("exact", "normalized", "symbolic")})
# by level, symbolic vs exact gap (off r0)
lv = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r["setting"] == "off" and r["rep"] == "r0":
        lv[r["level"]][0] += 1; lv[r["level"]][1] += r["exact"]; lv[r["level"]][2] += r["symbolic"]
out["by_level_off_r0"] = {k: v for k, v in sorted(lv.items())}
json.dump(out, open(R + "math_results.json", "w"), indent=1, default=str)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else json.dumps(v, default=str)[:1500])
