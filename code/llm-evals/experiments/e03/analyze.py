"""Part 3 analysis: every number in the article comes from here. Writes results.json."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import stats as S  # noqa: E402

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
ITEMS = json.loads(Path("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json").read_text())
IDS = [it["id"] for it in ITEMS]
SUBJ = np.array([it["subject"] for it in ITEMS])
R = {}


def answer_of(text, stop="end_turn"):
    """The reply's answer: a letter A-D at the very start, followed by nothing, a space or
    punctuation ("B", "B.", "A\n\nStatement 1 ..."); or, if the reply finished (not cut off at
    max_tokens), a lone letter or "Answer: X" on its last line ("5144/8 = 643\n\nB").
    Anything else has no answer and is scored wrong."""
    text = text or ""
    m = re.match(r"\s*([ABCD])(?:[\s.):]|$)", text)
    if m:
        return m.group(1)
    if stop == "end_turn" and text.strip():
        last = text.strip().splitlines()[-1].strip()
        m = re.fullmatch(r"(?:\**answer\**:?\s*)?\**\(?([ABCD])\)?\.?\**", last, re.I)
        if m:
            return m.group(1).upper()
    return None


def matrix(config):
    """(items, K) 0/1 matrix in ITEMS order; also non-letter count and cost."""
    rows = [json.loads(l) for l in (RUNS / f"{config}.jsonl").read_text().splitlines()]
    by = defaultdict(dict)
    nolet, cost, errors, out_tok, think = 0, 0.0, 0, [], 0
    for r in rows:
        if "error" in r:
            errors += 1
            continue
        pred = answer_of(r["text"], r["stop"])
        by[r["id"]][r["k"]] = int(pred == r["answer"])
        nolet += pred is None
        cost += r["cost"]
        out_tok.append(r["usage"]["output_tokens"])
        think += r["thinking_blocks"] > 0
    K = max(len(v) for v in by.values())
    X = np.array([[by[i][k] for k in range(K)] for i in IDS], float)
    meta = dict(requests=len(rows), errors=errors, not_single_letter=nolet, cost=cost,
                mean_output_tokens=float(np.mean(out_tok)), share_with_thinking=think / len(out_tok))
    return X, meta


def r4(x):
    return round(float(x), 4)


def ci_block(s, clusters=SUBJ, binary=True):
    m, se = S.mean_se(s)
    out = dict(mean=r4(m), se_clt=r4(se), ci_clt=[r4(m - 1.96 * se), r4(m + 1.96 * se)])
    if binary:
        out["ci_wilson"] = [r4(v) for v in S.wilson(int(s.sum()), len(s))]
    sc = S.clustered_se(s, clusters)
    out.update(se_clustered=r4(sc), ci_clustered=[r4(m - 1.96 * sc), r4(m + 1.96 * sc)],
               design_effect=r4((sc / se) ** 2),
               ci_boot_items=[r4(v) for v in S.bootstrap_ci(s, n_boot=10000)],
               ci_boot_subjects=[r4(v) for v in S.bootstrap_ci(s, clusters=clusters, n_boot=10000)])
    return out


# ---------------------------------------------------------------- Experiment 1 -----------------
H, R["haiku_off_meta"] = matrix("haiku-off")
e1 = R["exp1"] = {}
e1["single_run_k0"] = ci_block(H[:, 0])
e1["each_run"] = [r4(H[:, k].mean()) for k in range(H.shape[1])]
e1["mean_of_5"] = ci_block(H.mean(1), binary=False)
vs = S.variance_split(H)
e1["variance_split"] = {k: (r4(v) if isinstance(v, float) else v) for k, v in vs.items()}
c = H.sum(1)
e1["items_by_correct_count"] = {int(j): int((c == j).sum()) for j in range(H.shape[1] + 1)}
subj_acc = {s: float(H[SUBJ == s].mean()) for s in np.unique(SUBJ)}
lo, hi = min(subj_acc, key=subj_acc.get), max(subj_acc, key=subj_acc.get)
e1["subjects"] = dict(lowest=[lo, r4(subj_acc[lo])], highest=[hi, r4(subj_acc[hi])],
                      n_below_70=sum(v < 0.7 for v in subj_acc.values()),
                      n_at_100=sum(v == 1 for v in subj_acc.values()))
# share of variance between subjects (one-way ANOVA style, on per-item rates)
p_item = H.mean(1)
subj_means = np.array([subj_acc[s] for s in SUBJ])
e1["share_item_variance_between_subjects"] = r4(subj_means.var() / p_item.var())

# CI width vs n: subsample items without replacement, 300 draws per n
rng = np.random.default_rng(1)
curve = []
for n in [20, 50, 100, 200, 400, 800, 1140]:
    w1, w5, wc = [], [], []
    for _ in range(300 if n < 1140 else 1):
        idx = rng.choice(len(IDS), n, replace=False)
        s1 = H[idx, rng.integers(0, 5)]
        lo_, hi_ = S.wilson(int(s1.sum()), n)
        w1.append((hi_ - lo_) / 2)
        w5.append(1.96 * S.mean_se(H[idx].mean(1))[1])
        wc.append(1.96 * S.clustered_se(s1, SUBJ[idx]))
    curve.append(dict(n=n, wilson_1=r4(np.mean(w1)), clt_5=r4(np.mean(w5)), clustered_1=r4(np.mean(wc))))
e1["width_vs_n"] = curve

# Coverage check by resampling: the 1,140 x 5 answers are the "population"; its mean is the truth.
truth = H.mean()
cov = []
for n in [10, 20, 30, 50, 100, 200]:
    hits = defaultdict(int)
    sims = 4000
    for _ in range(sims):
        idx = rng.integers(0, len(IDS), n)
        s1 = H[idx, rng.integers(0, 5, n)]
        m, se = s1.mean(), s1.std(ddof=1) / np.sqrt(n)
        hits["wald"] += (m - 1.96 * se <= truth <= m + 1.96 * se)
        lo_, hi_ = S.wilson(int(s1.sum()), n)
        hits["wilson"] += lo_ <= truth <= hi_
        b = s1[rng.integers(0, n, (400, n))].mean(1)
        blo, bhi = np.quantile(b, [0.025, 0.975])
        hits["bootstrap"] += blo <= truth <= bhi
    cov.append(dict(n=n, **{k: r4(v / sims) for k, v in hits.items()}))
e1["coverage_items"] = dict(truth=r4(truth), rows=cov)

# Clustered sampling: draw m whole subjects (20 items each), one sample per item.
subjects = np.unique(SUBJ)
groups = [np.nonzero(SUBJ == s)[0] for s in subjects]
covc = []
for m_ in [5, 10, 20, 40]:
    hits = defaultdict(int)
    sims = 4000
    for _ in range(sims):
        pick = rng.integers(0, len(groups), m_)
        idx = np.concatenate([groups[j] for j in pick])
        lab = np.repeat(np.arange(m_), 20)
        s1 = H[idx, rng.integers(0, 5, len(idx))]
        mu = s1.mean()
        lo_, hi_ = S.wilson(int(s1.sum()), len(s1))
        hits["wilson_naive"] += lo_ <= truth <= hi_
        sc = S.clustered_se(s1, lab)
        hits["clustered"] += mu - 1.96 * sc <= truth <= mu + 1.96 * sc
    covc.append(dict(subjects=m_, items=20 * m_, **{k: r4(v / sims) for k, v in hits.items()}))
e1["coverage_subjects"] = covc

# ---------------------------------------------------------------- Experiment 2 -----------------
systems = {"haiku-off": H}
for cfg in ["haiku-low", "sonnet-off"]:
    if (RUNS / f"{cfg}.jsonl").exists():
        X, R[f"{cfg}_meta"] = matrix(cfg)
        systems[cfg] = X
        R[f"{cfg}_acc"] = ci_block(X[:, 0])
        R[f"{cfg}_acc_meanK"] = ci_block(X.mean(1), binary=False)
        R[f"{cfg}_each_run"] = [r4(X[:, k].mean()) for k in range(X.shape[1])]
        R[f"{cfg}_variance_split"] = {k: (r4(v) if isinstance(v, float) else v)
                                      for k, v in S.variance_split(X).items()} if X.shape[1] > 1 else None


def compare(A, B, name):
    a, b = A[:, 0], B[:, 0]
    pdiff = S.paired_diff(a, b, clusters=SUBJ)
    mc = S.mcnemar(a, b)
    t, pt = S.paired_t(a, b)
    out = dict(single=dict(
        diff=r4(pdiff["diff"]), corr=r4(pdiff["corr"]),
        se_unpaired=r4(pdiff["se_unpaired"]), se_paired=r4(pdiff["se_paired"]),
        se_paired_clustered=r4(pdiff["se_paired_clustered"]),
        ci_unpaired=[r4(v) for v in pdiff["ci_unpaired"]], ci_paired=[r4(v) for v in pdiff["ci_paired"]],
        ci_paired_clustered=[r4(v) for v in pdiff["ci_paired_clustered"]],
        ci_paired_bootstrap=[r4(v) for v in S.paired_bootstrap(a, b)],
        mcnemar=mc, paired_t=[r4(t), float(pt)], sign_flip_p=S.sign_flip_p(a, b)))
    am, bm = A.mean(1), B.mean(1)
    pk = S.paired_diff(am, bm, clusters=SUBJ)
    out["meanK"] = dict(KA=A.shape[1], KB=B.shape[1], diff=r4(pk["diff"]), corr=r4(pk["corr"]),
                        ci_unpaired=[r4(v) for v in pk["ci_unpaired"]], ci_paired=[r4(v) for v in pk["ci_paired"]],
                        ci_paired_clustered=[r4(v) for v in pk["ci_paired_clustered"]],
                        sign_flip_p=S.sign_flip_p(am, bm))
    # power: Miller's eq. 9 from measured variance components, then a resampling check
    pin = S.power_inputs(A, B) if min(A.shape[1], B.shape[1]) > 1 else None
    delta = abs(B.mean() - A.mean())
    pw = dict(delta=r4(delta))
    if pin:
        pw.update({k: r4(v) for k, v in pin.items()})
        for K in [1, 5]:
            pw[f"n_paired_K{K}"] = round(S.sample_size(delta, pin["omega2"], pin["sigma2_a"], pin["sigma2_b"], K, K))
        pw["n_paired_Kinf"] = round(S.sample_size(delta, pin["omega2"]))
        pw["mde_1140_K1"] = r4(S.mde(1140, pin["omega2"], pin["sigma2_a"], pin["sigma2_b"]))
    va, vb = A.mean() * (1 - A.mean()), B.mean() * (1 - B.mean())
    pw["n_unpaired_K1"] = round(S.unpaired_sample_size(delta, va, vb))
    pw["mde_1140_unpaired_K1"] = r4(1 / np.sqrt(1140) * (1.959964 + 0.841621) * np.sqrt(va + vb))
    checks = {}
    if pin and pw["n_paired_K1"] <= 5000:
        n_p = pw["n_paired_K1"]
        checks["paired_at_n_paired"] = S.resampled_power(A, B, n_p, k=1, paired=True, n_sim=3000, seed=2)
        checks["unpaired_at_n_paired"] = S.resampled_power(A, B, n_p, k=1, paired=False, n_sim=3000, seed=3)
    n_u = pw["n_unpaired_K1"]
    if n_u <= 5000:
        checks["unpaired_at_n_unpaired"] = S.resampled_power(A, B, n_u, k=1, paired=False, n_sim=1500, seed=4)
    pw["resampling_checks"] = {k: r4(v) for k, v in checks.items()}
    # power curves for the chart
    grid = [50, 100, 200, 400, 800, 1140, 2000, 4000]
    pw["curve"] = [dict(n=n, paired=r4(S.resampled_power(A, B, n, 1, True, 1500, seed=10 + n)),
                        unpaired=r4(S.resampled_power(A, B, n, 1, False, 1500, seed=20 + n))) for n in grid]
    out["power"] = pw
    R[f"compare_{name}"] = out


if "haiku-low" in systems:
    compare(H, systems["haiku-low"], "off_vs_low")
if "sonnet-off" in systems:
    compare(H, systems["sonnet-off"], "haiku_vs_sonnet")
if "sonnet-off" in systems and "haiku-low" in systems:
    compare(systems["haiku-low"], systems["sonnet-off"], "low_vs_sonnet")

# ---------------------------------------------------------------- Experiment 3 -----------------
qpath = RUNS / "qwen-1.5b.jsonl"
if qpath.exists():
    q = {json.loads(l)["id"]: json.loads(l) for l in qpath.read_text().splitlines()}
    Q = np.array([[q[i]["correct"]] for i in IDS], float)
    Qp = np.array([q[i]["p_correct"] for i in IDS])
    systems["qwen-1.5b"] = Q
    R["qwen_acc"] = ci_block(Q[:, 0])
    R["qwen_expected"] = ci_block(Qp, binary=False)  # Miller rec. 3: score with the probability
    R["qwen_mass_abcd_median"] = r4(np.median([q[i]["mass_abcd"] for i in IDS]))

names = list(systems)
pairs, pv = [], []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = systems[names[i]][:, 0], systems[names[j]][:, 0]
        mc = S.mcnemar(a, b)
        pairs.append(dict(a=names[i], b=names[j], diff=r4(b.mean() - a.mean()), **mc))
        pv.append(mc["p"])
if pv:
    for name, fn in [("bonferroni", S.bonferroni), ("holm", S.holm), ("bh", S.benjamini_hochberg)]:
        for p_, rej in zip(pairs, fn(pv)):
            p_[name] = bool(rej)
    for p_ in pairs:
        p_["raw_05"] = p_["p"] < 0.05
R["pairs"] = pairs

# Null leaderboard: the five Haiku runs as five "models" that are truly identical.
null = []
for i in range(5):
    for j in range(i + 1, 5):
        mc = S.mcnemar(H[:, i], H[:, j])
        null.append(dict(a=i, b=j, diff=r4(H[:, j].mean() - H[:, i].mean()), p=r4(mc["p"])))
R["null_real_5runs"] = null

# Simulated: 20 identical "models", each one sample per item drawn from Haiku's measured per-item rates.
rng = np.random.default_rng(7)
sims, M = 1000, 20
fp_raw, fp_holm, any_raw, any_holm, gaps = [], [], 0, 0, []
p_i = H.mean(1)
for _ in range(sims):
    Y = (rng.random((M, len(p_i))) < p_i).astype(int)
    scores = Y.mean(1)
    gaps.append(scores.max() - scores.min())
    ps = [S.mcnemar(Y[a], Y[b])["p"] for a in range(M) for b in range(a + 1, M)]
    raw = np.array(ps) < 0.05
    hol = S.holm(ps)
    fp_raw.append(raw.sum())
    fp_holm.append(hol.sum())
    any_raw += raw.any()
    any_holm += hol.any()
R["null_sim_20"] = dict(models=M, pairs=M * (M - 1) // 2, sims=sims,
                        mean_false_pos_raw=r4(np.mean(fp_raw)), mean_false_pos_holm=r4(np.mean(fp_holm)),
                        share_any_raw=r4(any_raw / sims), share_any_holm=r4(any_holm / sims),
                        gap_median=r4(np.median(gaps)), gap_p95=r4(np.quantile(gaps, 0.95)))

(ROOT / "results.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1)[:6000])
