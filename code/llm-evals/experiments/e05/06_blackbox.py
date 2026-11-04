"""Experiment 3: black-box contamination probes on Claude (Message Batches, through the ledger).
  python 06_blackbox.py jobs <model> <n>    # build and print the job count + estimate
  python 06_blackbox.py run <model> <n>     # submit
Two probes, each paired with a control:
  guided completion (Golchin and Surdeanu 2023) on ARC-Challenge: the first part of a question,
    with the dataset named ("guided") or not ("general"); and the guided prompt on the first part
    of the Haiku paraphrase of the same question (text that did not exist before November 2026).
  TS-Guessing (Deng et al. 2023) on MMLU (mmlu-1140) and ARC-Challenge: one wrong option hidden.
Thinking off ({"type": "disabled"} on Haiku, {"type": "between_tools"} on Sonnet).
"""
import json, sys
import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
DATA = "/home/amit/.cache/vidyalai-scratch/evals/data"
mode, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
short = "haiku" if "haiku" in model else "sonnet"

SYS_GUIDED = ("You are given the first part of a question from the test split of the {ds} dataset. "
              "Complete it exactly as it appears in that dataset. Reply with the rest of the question "
              "only, starting from the next word.")
SYS_GENERAL = ("You are given the first part of a question. Complete it so that it makes a sensible "
               "question. Reply with the rest of the question only, starting from the next word.")
SYS_TS = ("Here is a multiple-choice question from the test set of the {ds} benchmark. One of the "
          "incorrect options has been replaced by [MASK]. Fill in [MASK] with the exact text of that "
          "option as it appears in the benchmark. Reply with the option text only.")
SYS_SOFT = ("This is the beginning of a question from the {ds} test set. Finish the question. "
            "Reply with the rest of the question only, starting from the next word.")
SYS_TS_SOFT = ("This multiple-choice question is from the {ds} test set. One of its incorrect options "
               "is hidden as [MASK]. What was that option? Reply with the option text only.")
DS = {"arc": "ARC-Challenge (AI2 Reasoning Challenge)", "mmlu": "MMLU (Measuring Massive Multitask Language Understanding)"}


def split(q, rng):
    w = q.split()
    k = int(round(len(w) * rng.uniform(0.4, 0.6)))
    k = min(max(k, 1), len(w) - 1)
    return " ".join(w[:k]), " ".join(w[k:])


def p(system, user, max_tokens):
    return {"model": model, "max_tokens": max_tokens, "system": system,
            "thinking": C.THINK_OFF[model], "messages": [{"role": "user", "content": user}]}


arc = json.load(open(f"{DATA}/e01-arc-challenge.json"))
mmlu = json.load(open(f"{DATA}/mmlu-1140.json"))
para = {int(json.loads(l)["id"][1:]): json.loads(l)["text"].strip() for l in open(f"{RUNS}/paraphrase-clean.jsonl")}
rng = np.random.default_rng(5)
arc_pick = [int(i) for i in rng.permutation(len(arc)) if int(i) in para and len(arc[int(i)]["question"].split()) >= 8][:n]
mmlu_pick = [int(i) for i in rng.permutation(len(mmlu)) if len(mmlu[int(i)]["question"].split()) >= 4][:n]

jobs, truth = {}, {}
srng = np.random.default_rng(6)
for i in arc_pick:
    head, tail = split(arc[i]["question"], srng)
    jobs[f"gc-guided-{i}"] = p(SYS_GUIDED.format(ds=DS["arc"]), head, 150); truth[f"gc-guided-{i}"] = tail
    jobs[f"gc-general-{i}"] = p(SYS_GENERAL, head, 150); truth[f"gc-general-{i}"] = tail
    jobs[f"gc-soft-{i}"] = p(SYS_SOFT.format(ds=DS["arc"]), head, 150); truth[f"gc-soft-{i}"] = tail
    ph, pt = split(para[i], srng)
    jobs[f"gc-para-{i}"] = p(SYS_SOFT.format(ds=DS["arc"]), ph, 150); truth[f"gc-para-{i}"] = pt
mrng = np.random.default_rng(7)
for ds, pool, pick in (("arc", arc, arc_pick), ("mmlu", mmlu, mmlu_pick)):
    for i in pick:
        it = pool[i]
        wrong = [j for j in range(len(it["choices"])) if j != it["answer"]]
        m = int(mrng.choice(wrong))
        opts = "\n".join(f"{'ABCDE'[j]}. {'[MASK]' if j == m else c}" for j, c in enumerate(it["choices"]))
        for kind, sysp in (("strict", SYS_TS), ("soft", SYS_TS_SOFT)):
            key = f"ts{kind}-{ds}-{i}"
            jobs[key] = p(sysp.format(ds=DS[ds]), f"{it['question']}\n\n{opts}", 60)
            truth[key] = it["choices"][m]
json.dump(truth, open(f"{RUNS}/blackbox-truth-{short}.json", "w"))
est = sum(__import__("ledger").estimate_cost(model, len(json.dumps(v)) // 2, v["max_tokens"]) for v in jobs.values()) * 0.5
print(len(jobs), "jobs, worst-case estimate $%.4f" % est)
if mode == "run":
    print(C.run_batch("e05", jobs, f"{RUNS}/blackbox-{short}.jsonl", f"e05 black-box probes {short}"))
if mode == "pilot":
    import ledger
    cl = ledger.client()
    keys = [k for k in jobs if k.startswith(("gc-soft", "gc-para", "tssoft-arc", "tssoft-mmlu"))]
    keys = [k for pre in ("gc-soft", "gc-para", "tssoft-arc", "tssoft-mmlu") for k in keys if k.startswith(pre)][:0] or \
           [k for pre in ("gc-soft", "gc-para", "tssoft-arc", "tssoft-mmlu") for k in [x for x in jobs if x.startswith(pre)][:2]]
    with open(f"{RUNS}/blackbox-pilot2-{short}.jsonl", "w") as f:
        for k in keys:
            r = ledger.create(cl, "e05", note=f"e05 black-box pilot {short}", **jobs[k])
            t = "".join(c.text for c in r.content if c.type == "text")
            f.write(json.dumps({"id": k, "text": t, "truth": truth[k], "usage": r.usage.model_dump(), "stop": r.stop_reason}) + "\n")
            print(k, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, "\n  OUT:", t[:200], "\n  TRUE:", truth[k][:200])
