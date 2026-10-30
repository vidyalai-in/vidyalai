# Experiment A, second pass on Gemma 2 2B IT:
#  (1) harder city facts (1,395 cities of 100k-500k people): neutral, "wrong" (Sharma et al. template)
#      and "strong" (authority) prompts; last-token residual stream at 14 layers + A/B margin.
#  (2) opinion sycophancy on Anthropic's model-written evals (CC-BY-4.0): 500 items from each of the
#      NLP-survey, PhilPapers (two-choice items only) and political-typology sets; A/B margin only.
import json, random, time, numpy as np, torch
from lab import safety as S
from collect_it import Q, PREFILL, SYC, LAYERS

OUT = "/home/amit/.cache/vidyalai-scratch/p11/data/it"
MWE = "/home/amit/.cache/vidyalai-scratch/interp/data/p11/mwe"


def mwe_items(n=500, seed=0):
    out = {}
    for name in ["nlp_survey", "philpapers2020", "political_typology_quiz"]:
        rows = [json.loads(l) for l in open(f"{MWE}/sycophancy_on_{name}.jsonl")]
        rows = [r for r in rows if isinstance(r["answer_not_matching_behavior"], str)
                or len(r["answer_not_matching_behavior"]) == 1]
        rows = [r for r in rows if "(C)" not in r["question"]]
        random.Random(seed).shuffle(rows)
        out[name] = rows[:n]
    return out


def mwe_chat(q):
    q = q.rstrip()
    if q.endswith("Answer:"):
        q = q[: -len("Answer:")].rstrip()
    return S.chat(q, PREFILL)


if __name__ == "__main__":
    hard = json.load(open("/home/amit/.cache/vidyalai-scratch/p11/data/hard_cities.json"))
    meta, P = [], {k: [] for k in ["neutral", "wrong", "strong"]}
    for ci, it in enumerate(hard):
        for order in (0, 1):
            a, b = (it["true"], it["false"]) if order == 0 else (it["false"], it["true"])
            q = Q.format(c=it["city"], a=a, b=b)
            meta.append({"city": it["city"], "ci": ci, "order": order})
            for k in P:
                s = SYC[k].format(t=it["true"], f=it["false"])
                P[k].append(S.chat(q + ("\n" + s if s else ""), PREFILL))
    t0 = time.time()
    model, tok = S.load_gemma(S.IT)
    keep = list(S.letter_ids(tok))
    order = np.array([m["order"] for m in meta])
    res = {"meta": meta}
    for k, ps in P.items():
        t1 = time.time()
        H, lg = S.last_token_states(model, tok, ps, layers=LAYERS, batch=48, keep_ids=keep)
        np.save(f"{OUT}/Hhard_{k}.npy", H)
        m = S.ab_margin(tok, lg)
        res[k] = np.where(order == 0, m, -m).tolist()
        print("hard", k, len(ps), "s", round(time.time() - t1, 1), "acc", round(float((np.array(res[k]) > 0).mean()), 3),
              "top A/B", round(float(np.isin(lg[:, -1].numpy(), keep).mean()), 3), flush=True)
    json.dump(res, open(f"{OUT}/hard_margins.json", "w"))
    mw = mwe_items()
    out = {}
    for name, rows in mw.items():
        t1 = time.time()
        _, lg = S.last_token_states(model, tok, [mwe_chat(r["question"]) for r in rows], layers=[0], batch=16, keep_ids=keep)
        m = S.ab_margin(tok, lg)          # A minus B
        match_is_a = np.array([r["answer_matching_behavior"].strip() == "(A)" for r in rows])
        out[name] = {"margin_A_minus_B": m.tolist(), "match_is_A": match_is_a.tolist(),
                     "top_ab": float(np.isin(lg[:, -1].numpy(), keep).mean())}
        syc = np.where(match_is_a, m > 0, m < 0).mean()
        print("mwe", name, len(rows), "s", round(time.time() - t1, 1), "matches user", round(float(syc), 3),
              "top A/B", out[name]["top_ab"], flush=True)
    json.dump(out, open(f"{OUT}/mwe_it.json", "w"))
    print("total s", round(time.time() - t0, 1), "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
