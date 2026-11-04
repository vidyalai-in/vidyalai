"""Build the two working sets for Part 6 (kept in e06/data, never committed: third-party text).

summeval.json  100 CNN/DailyMail articles x 16 systems, 3 expert ratings per summary on four
               dimensions (SummEval, Fabbri et al. 2021, MIT), the original reference highlight
               plus 10 crowd-written references, and the article text (mteb/summeval, joined by id).
mtbench.json   MT-Bench human pairwise votes on turn 1 (lmsys/mt_bench_human_judgments,
               CC-BY-4.0): one record per unique (question, model pair), all its votes, both
               answers, and GPT-4's released pairwise verdict for the same pair.
"""
import json
from collections import defaultdict

from datasets import load_dataset

D = "/home/amit/.cache/vidyalai-scratch/e06/data/"

# ---- SummEval
arts = {r["id"]: r["text"] for r in load_dataset("mteb/summeval")["test"]}
rows = [json.loads(l) for l in open(D + "summeval_annotations.aligned.jsonl")]
docs = defaultdict(dict)
for r in rows:
    docs[r["id"]][r["model_id"]] = r
systems = sorted({r["model_id"] for r in rows}, key=lambda s: int(s[1:]))
out = []
for did in sorted(docs):
    d = docs[did]
    refs = d[systems[0]]["references"]
    assert all(d[s]["references"] == refs for s in systems)
    out.append(dict(id=did, article=arts[did], refs=refs, systems=systems,
                    summaries=[d[s]["decoded"] for s in systems],
                    expert=[d[s]["expert_annotations"] for s in systems]))
json.dump(out, open(D + "summeval.json", "w"))
print("summeval", len(out), "docs x", len(systems), "systems")

# ---- MT-Bench, turn 1
mt = load_dataset("lmsys/mt_bench_human_judgments")
qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(D + "mtbench_question.jsonl")}
refs = {json.loads(l)["question_id"]: json.loads(l)["choices"][0]["turns"][0] for l in open(D + "mtbench_ref_gpt4.jsonl")}


def vote(winner, flip):
    w = {"model_a": "A", "model_b": "B"}.get(winner, "tie")
    return {"A": "B", "B": "A"}.get(w, w) if flip else w


pairs = {}
for r in mt["human"]:
    if r["turn"] != 1:
        continue
    m1, m2 = sorted([r["model_a"], r["model_b"]])
    key = f'{r["question_id"]}|{m1}|{m2}'
    flip = r["model_a"] != m1
    ans = {r["model_a"]: r["conversation_a"][1]["content"], r["model_b"]: r["conversation_b"][1]["content"]}
    p = pairs.setdefault(key, dict(key=key, qid=r["question_id"], category=qs[r["question_id"]]["category"],
                                   question=r["conversation_a"][0]["content"], model_1=m1, model_2=m2,
                                   answer_1=ans[m1], answer_2=ans[m2], reference=refs.get(r["question_id"]),
                                   votes=[], gpt4=None))
    assert p["answer_1"] == ans[m1] and p["answer_2"] == ans[m2]
    # votes are stored from model_1's side: "A" = model_1 better, "B" = model_2 better
    p["votes"].append(dict(judge=r["judge"], vote=vote(r["winner"], flip)))
n_gpt = 0
for r in mt["gpt4_pair"]:
    if r["turn"] != 1:
        continue
    m1, m2 = sorted([r["model_a"], r["model_b"]])
    key = f'{r["question_id"]}|{m1}|{m2}'
    if key in pairs:
        w = r["winner"]
        v = "tie" if w.startswith("tie") else vote(w, r["model_a"] != m1)
        pairs[key]["gpt4"] = dict(vote=v, inconsistent=(w == "tie (inconsistent)"))
        n_gpt += 1
out = sorted(pairs.values(), key=lambda p: p["key"])
json.dump(out, open(D + "mtbench.json", "w"))
print("mtbench turn-1 pairs", len(out), "votes", sum(len(p["votes"]) for p in out), "with gpt4", n_gpt)
