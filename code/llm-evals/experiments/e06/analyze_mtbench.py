"""Experiments 2 and 3: pairwise judges on MT-Bench turn-1 human votes.
Writes results/mtbench.json (small, committable)."""
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import grading as G
import stats as ST
G.wilson = ST.wilson

D = "/home/amit/.cache/vidyalai-scratch/e06/"
mt = json.load(open(D + "data/mtbench.json"))
FLIP = {"A": "B", "B": "A", "tie": "tie", None: None}


def load(run):
    p = Path(D + f"runs/{run}.jsonl")
    if not p.exists():
        return None
    r = G.load_runs(p)
    rp = Path(D + f"runs/{run}.retry.jsonl")
    if rp.exists():
        r.update(G.load_runs(rp))
    return r


def verdicts(recs, prefix=""):
    """{pair index: (v_ab, v_ba)} from model_1's side ('A' = model_1 better). v_ba is flipped back."""
    out = {}
    for cid, r in recs.items():
        parts = cid.split("-")
        i, order = int(parts[0][1:]), parts[-1]
        if prefix and parts[1] != prefix:
            continue
        if not prefix and len(parts) != 2:
            continue
        v = G.parse_verdict(r.get("text"))
        out.setdefault(i, [None, None])[0 if order == "ab" else 1] = v if order == "ab" else FLIP[v]
    return {i: tuple(v) for i, v in out.items()}


def combine(v):
    """MT-Bench convention: a win counts only if both orders agree; otherwise a tie."""
    a, b = v
    if a is None or b is None:
        return None
    return a if a == b else "tie"


def judge_vs_votes(jv, idxs, with_ties):
    """Per-vote agreement of a judge's verdict with each human vote; groups = question id."""
    xs, ys, gs = [], [], []
    for i in idxs:
        if jv.get(i) is None:
            continue
        for v in mt[i]["votes"]:
            xs.append(jv[i]); ys.append(v["vote"]); gs.append(mt[i]["qid"])
    xs, ys, gs = np.array(xs), np.array(ys), np.array(gs)
    keep = np.ones(len(xs), bool) if with_ties else (xs != "tie") & (ys != "tie")

    def fn(ix):
        ix = ix[keep[ix]]
        return (xs[ix] == ys[ix]).mean()
    est = fn(np.arange(len(xs)))
    lo, hi = G.boot_stat(fn, len(xs), groups=gs)
    return dict(est=float(est), lo=lo, hi=hi, n=int(keep.sum()))


def human_human(idxs, with_ties):
    xs, ys, gs = [], [], []
    for i in idxs:
        for u, w in itertools.combinations(mt[i]["votes"], 2):
            if u["judge"] == w["judge"]:
                continue
            xs.append(u["vote"]); ys.append(w["vote"]); gs.append(mt[i]["qid"])
    xs, ys, gs = np.array(xs), np.array(ys), np.array(gs)
    keep = np.ones(len(xs), bool) if with_ties else (xs != "tie") & (ys != "tie")

    def fn(ix):
        ix = ix[keep[ix]]
        return (xs[ix] == ys[ix]).mean()
    est = fn(np.arange(len(xs)))
    lo, hi = G.boot_stat(fn, len(xs), groups=gs)
    return dict(est=float(est), lo=lo, hi=hi, n=int(keep.sum()))


def kappa_votes(jv, idxs):
    xs, ys, gs = [], [], []
    for i in idxs:
        if jv.get(i) is None:
            continue
        for v in mt[i]["votes"]:
            xs.append(jv[i]); ys.append(v["vote"]); gs.append(mt[i]["qid"])
    xs, ys = np.array(xs), np.array(ys)
    est = G.cohen_kappa(xs, ys, ["A", "B", "tie"])
    lo, hi = G.boot_stat(lambda ix: G.cohen_kappa(xs[ix], ys[ix], ["A", "B", "tie"]), len(xs), groups=gs)
    return dict(est=float(est), lo=lo, hi=hi, n=len(xs))


def kappa_hh(idxs):
    xs, ys, gs = [], [], []
    for i in idxs:
        for u, w in itertools.combinations(mt[i]["votes"], 2):
            if u["judge"] != w["judge"]:
                xs.append(u["vote"]); ys.append(w["vote"]); gs.append(mt[i]["qid"])
    xs, ys = np.array(xs), np.array(ys)
    est = G.cohen_kappa(xs, ys, ["A", "B", "tie"])
    lo, hi = G.boot_stat(lambda ix: G.cohen_kappa(xs[ix], ys[ix], ["A", "B", "tie"]), len(xs), groups=gs)
    return dict(est=float(est), lo=lo, hi=hi, n=len(xs))


def swap_stats(V, idxs):
    """Consistency under swapping, and where inconsistent, which slot won."""
    n = cons = first = second = 0
    flags = []
    for i in idxs:
        a, b = V.get(i, (None, None))
        if a is None or b is None:
            continue
        n += 1
        c = a == b
        cons += c
        flags.append(1.0 if c else 0.0)
        if not c:
            # a is from the A-slot holding model_1; b (flipped back) from the slot where model_1 was B.
            # "first slot won" in order ab means a == 'A'; in order ba the first slot held model_2, i.e. b == 'B'.
            first += (a == "A") + (b == "B")
            second += (a == "B") + (b == "A")
    lo, hi = G.wilson(cons, n) if n else (np.nan, np.nan)
    return dict(n=n, consistent=cons / n if n else np.nan, lo=lo, hi=hi,
                inconsistent_first_slot_wins=first, inconsistent_second_slot_wins=second)


res = {}
all_idx = list(range(len(mt)))
multi = [i for i, p in enumerate(mt) if len({v["judge"] for v in p["votes"]}) > 1]
res["n_pairs"], res["n_votes"], res["n_multi_pairs"] = len(mt), sum(len(p["votes"]) for p in mt), len(multi)
res["human_vote_mix"] = dict(Counter(v["vote"] for p in mt for v in p["votes"]))

# human-human
res["human_human"] = {"S1": human_human(multi, True), "S2": human_human(multi, False), "kappa": kappa_hh(multi)}

# GPT-4's released judgments (both orders, inconsistent = tie)
gpt4 = {i: p["gpt4"]["vote"] for i, p in enumerate(mt) if p["gpt4"]}
res["gpt4_released"] = {"S1_all": judge_vs_votes(gpt4, all_idx, True), "S2_all": judge_vs_votes(gpt4, all_idx, False),
                        "S1_multi": judge_vs_votes(gpt4, multi, True), "S2_multi": judge_vs_votes(gpt4, multi, False),
                        "kappa_all": kappa_votes(gpt4, all_idx),
                        "inconsistent_share": float(np.mean([p["gpt4"]["inconsistent"] for p in mt]))}

# Haiku pairwise
pair = load("pair")
V = verdicts(pair)
H = {i: combine(v) for i, v in V.items()}
res["haiku_pair"] = {"S1_all": judge_vs_votes(H, all_idx, True), "S2_all": judge_vs_votes(H, all_idx, False),
                     "S1_multi": judge_vs_votes(H, multi, True), "S2_multi": judge_vs_votes(H, multi, False),
                     "kappa_all": kappa_votes(H, all_idx), "swap": swap_stats(V, all_idx),
                     "unparsed_after_retry": sum(v[0] is None or v[1] is None for v in V.values()),
                     "cutoffs_first_pass": sum(1 for r in G.load_runs(D + "runs/pair.jsonl").values() if r.get("stop") == "max_tokens"),
                     "verdict_mix": dict(Counter(H.values()))}
# single-order verdicts (no swap) for comparison
H_ab = {i: v[0] for i, v in V.items()}
res["haiku_pair"]["S2_all_one_order"] = judge_vs_votes(H_ab, all_idx, False)
res["haiku_pair"]["S1_all_one_order"] = judge_vs_votes(H_ab, all_idx, True)

# judge vs GPT-4 released
both = [i for i in all_idx if H.get(i) and i in gpt4]
res["haiku_vs_gpt4"] = dict(agree=float(np.mean([H[i] == gpt4[i] for i in both])), n=len(both),
                            kappa=G.cohen_kappa([H[i] for i in both], [gpt4[i] for i in both], ["A", "B", "tie"]))

# system level: each model's win rate (ties = half) from humans vs Haiku, over the same pairs
models = sorted({p["model_1"] for p in mt} | {p["model_2"] for p in mt})


def winrates(get, idxs):
    w, n = defaultdict(float), defaultdict(float)
    for i in idxs:
        p = mt[i]
        for v in get(i):
            s1 = {"A": 1.0, "B": 0.0, "tie": 0.5}[v]
            w[p["model_1"]] += s1; w[p["model_2"]] += 1 - s1
            n[p["model_1"]] += 1; n[p["model_2"]] += 1
    return np.array([w[m] / n[m] for m in models])


hum = lambda i: [v["vote"] for v in mt[i]["votes"]]
hai = lambda i: [H[i]] * len(mt[i]["votes"]) if H.get(i) else []
qids = np.array([mt[i]["qid"] for i in all_idx])
from scipy import stats as st
wh, wj = winrates(hum, all_idx), winrates(hai, all_idx)
rng = np.random.default_rng(0)
uq = np.unique(qids)
taus = []
for _ in range(2000):
    pick = rng.choice(uq, len(uq))
    idx = np.concatenate([np.flatnonzero(qids == q) for q in pick])
    taus.append(st.kendalltau(winrates(hum, idx), winrates(hai, idx))[0])
res["system"] = dict(models=models, human_winrate=wh.tolist(), haiku_winrate=wj.tolist(),
                     kendall=float(st.kendalltau(wh, wj)[0]),
                     kendall_ci=[float(x) for x in np.nanquantile(taus, [0.025, 0.975])])

# length: among decisive verdicts, how often the longer answer wins
def longer_wins(get, idxs):
    k = n = 0
    for i in idxs:
        p = mt[i]
        l1, l2 = len(p["answer_1"]), len(p["answer_2"])
        if l1 == l2:
            continue
        for v in get(i):
            if v == "tie":
                continue
            n += 1
            k += (v == "A") == (l1 > l2)
    return dict(share=k / n, lo=G.wilson(k, n)[0], hi=G.wilson(k, n)[1], n=n)


res["longer_wins"] = {"human": longer_wins(hum, all_idx), "haiku": longer_wins(lambda i: [H[i]] if H.get(i) else [], all_idx),
                      "gpt4": longer_wins(lambda i: [gpt4[i]], all_idx)}

# family check: win rate of each model under humans vs Haiku on the same pairs (per-vote weighting)
fam = {}
for m in models:
    idxs = [i for i in all_idx if m in (mt[i]["model_1"], mt[i]["model_2"]) and H.get(i)]
    def score(get, ix):
        s = []
        for i in ix:
            for v in get(i):
                s1 = {"A": 1.0, "B": 0.0, "tie": 0.5}[v]
                s.append(s1 if mt[i]["model_1"] == m else 1 - s1)
        return np.mean(s)
    gs = np.array([mt[i]["qid"] for i in idxs])
    diff = lambda ix: score(hai, [idxs[j] for j in ix]) - score(hum, [idxs[j] for j in ix])
    lo, hi = G.boot_stat(diff, len(idxs), groups=gs, n_boot=1000)
    fam[m] = dict(human=float(score(hum, idxs)), haiku=float(score(hai, idxs)), diff=float(diff(np.arange(len(idxs)))),
                  lo=lo, hi=hi, n_pairs=len(idxs))
res["per_model"] = fam

# verbosity
vb = load("verbose")
if vb:
    S = verdicts(vb, "self")
    # self-duel from the padded copy's side: in order ab the padded copy is B; in ba it is A.
    # verdicts() flipped ba back, so in both entries 'B' = padded copy wins, 'A' = original wins.
    cnt = {"ab": Counter(v[0] for v in S.values()), "ba": Counter(v[1] for v in S.values())}
    padded_win = [((v[0] == "B") + (v[1] == "B")) / 2 for v in S.values()]
    orig_win = [((v[0] == "A") + (v[1] == "A")) / 2 for v in S.values()]
    comb = Counter(combine(v) for v in S.values())
    res["verbosity_self"] = dict(n=len(S), per_order={k: dict(c) for k, c in cnt.items()},
                                 padded_win=float(np.mean(padded_win)),
                                 padded_win_ci=G.boot_stat(lambda ix: np.mean(np.array(padded_win)[ix]), len(S)),
                                 orig_win=float(np.mean(orig_win)),
                                 orig_win_ci=G.boot_stat(lambda ix: np.mean(np.array(orig_win)[ix]), len(S)),
                                 combined=dict(comb))
    P1, P2 = verdicts(vb, "pad1"), verdicts(vb, "pad2")
    idxs = sorted(set(P1) & set(P2) & set(V))
    s = lambda v: np.mean([{"A": 1.0, "B": 0.0, "tie": 0.5}[x] for x in v if x])  # model_1's score, both orders
    base = np.array([s(V[i]) for i in idxs])
    pad1 = np.array([s(P1[i]) for i in idxs])
    pad2 = np.array([s(P2[i]) for i in idxs])
    res["verbosity_pairs"] = dict(n=len(idxs), base=float(base.mean()), pad1=float(pad1.mean()), pad2=float(pad2.mean()),
                                  shift_pad1=float((pad1 - base).mean()),
                                  shift_pad1_ci=G.boot_stat(lambda ix: (pad1 - base)[ix].mean(), len(idxs)),
                                  shift_pad2=float((pad2 - base).mean()),
                                  shift_pad2_ci=G.boot_stat(lambda ix: (pad2 - base)[ix].mean(), len(idxs)),
                                  flipped_pad1=float(np.mean([combine(P1[i]) != combine(V[i]) for i in idxs])),
                                  flipped_pad2=float(np.mean([combine(P2[i]) != combine(V[i]) for i in idxs])),
                                  swap_pad1=swap_stats(P1, idxs), swap_base=swap_stats(V, idxs))
    res["verbosity_pairs"]["padded_side_gain"] = float(((pad1 - base) + (base - pad2)).mean() / 2)
    res["verbosity_pairs"]["padded_side_gain_ci"] = G.boot_stat(lambda ix: (((pad1 - base) + (base - pad2)) / 2)[ix].mean(), len(idxs))
    res["padding_words"] = len(G.PADDING.split())

# reference-guided
rf = load("ref")
if rf:
    R = {i: combine(v) for i, v in verdicts(rf).items()}
    idxs = sorted(R)
    res["reference"] = dict(n_pairs=len(idxs), categories=dict(Counter(mt[i]["category"] for i in idxs)),
                            with_ref_S1=judge_vs_votes(R, idxs, True), with_ref_S2=judge_vs_votes(R, idxs, False),
                            no_ref_S1=judge_vs_votes(H, idxs, True), no_ref_S2=judge_vs_votes(H, idxs, False),
                            gpt4_S2=judge_vs_votes(gpt4, idxs, False), gpt4_S1=judge_vs_votes(gpt4, idxs, True),
                            swap_ref=swap_stats(verdicts(rf), idxs), swap_noref=swap_stats(V, idxs))
    hh_idx = [i for i in idxs if i in multi]
    res["reference"]["human_human_S2"] = human_human(hh_idx, False)
    res["reference"]["human_human_S1"] = human_human(hh_idx, True)
    # paired difference in per-vote agreement (S1), clustered by question
    xs = []
    for i in idxs:
        for v in mt[i]["votes"]:
            xs.append(((R[i] == v["vote"]) - (H[i] == v["vote"]), mt[i]["qid"]))
    d = np.array([x[0] for x in xs], float); g = np.array([x[1] for x in xs])
    res["reference"]["diff_S1"] = dict(est=float(d.mean()), ci=G.boot_stat(lambda ix: d[ix].mean(), len(d), groups=g))

# single-answer grading -> pairwise
sg = load("single")
if sg:
    rating = {}
    for cid, r in sg.items():
        q, m = cid.split("-")
        rating[(int(q[1:]), models[int(m[1:])])] = G.parse_rating(r.get("text"))
    SV = {}
    for i, p in enumerate(mt):
        a, b = rating.get((p["qid"], p["model_1"])), rating.get((p["qid"], p["model_2"]))
        if a is None or b is None:
            continue
        SV[i] = "A" if a > b else "B" if b > a else "tie"
    res["single"] = dict(n_ratings=len(rating), unparsed=sum(v is None for v in rating.values()),
                         S1_all=judge_vs_votes(SV, all_idx, True), S2_all=judge_vs_votes(SV, all_idx, False),
                         tie_share=float(np.mean([v == "tie" for v in SV.values()])),
                         rating_hist=dict(Counter(rating.values())))
    # paired: pairwise (both orders) vs single on the same votes
    xs = []
    for i in all_idx:
        if i in SV and H.get(i):
            for v in mt[i]["votes"]:
                xs.append(((H[i] == v["vote"]) - (SV[i] == v["vote"]), mt[i]["qid"]))
    d = np.array([x[0] for x in xs], float); g = np.array([x[1] for x in xs])
    res["single"]["pair_minus_single_S1"] = dict(est=float(d.mean()), ci=G.boot_stat(lambda ix: d[ix].mean(), len(d), groups=g))

# Sonnet as a second judge
sn = load("sonnet")
if sn:
    SVv = verdicts(sn)
    SN = {i: combine(v) for i, v in SVv.items()}
    idxs = sorted(SN)
    both = [i for i in idxs if SN[i] and H.get(i)]
    res["sonnet"] = dict(n_pairs=len(idxs), S1=judge_vs_votes(SN, idxs, True), S2=judge_vs_votes(SN, idxs, False),
                         haiku_S1=judge_vs_votes(H, idxs, True), haiku_S2=judge_vs_votes(H, idxs, False),
                         gpt4_S1=judge_vs_votes(gpt4, idxs, True), gpt4_S2=judge_vs_votes(gpt4, idxs, False),
                         swap=swap_stats(SVv, idxs), haiku_swap=swap_stats(V, idxs),
                         sonnet_haiku_agree=float(np.mean([SN[i] == H[i] for i in both])),
                         sonnet_haiku_kappa=G.cohen_kappa([SN[i] for i in both], [H[i] for i in both], ["A", "B", "tie"]),
                         sonnet_gpt4_agree=float(np.mean([SN[i] == gpt4[i] for i in both])),
                         cutoffs=sum(1 for r in sn.values() if r.get("stop") == "max_tokens"))
    lo, hi = G.wilson(sum(SN[i] == H[i] for i in both), len(both))
    res["sonnet"]["sonnet_haiku_agree_ci"] = [lo, hi]

json.dump(res, open(D + "results/mtbench.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:12000])
