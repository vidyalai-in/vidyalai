"""Side effects of the refusal-direction edits: FineWeb-Edu loss, ARC-Easy accuracy, and a strength sweep.
usage: side_effects.py <model_path> <tag> <part: static|sweep>"""
import sys, json, time, random
import numpy as np, torch
import pyarrow.parquet as pq
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.steering import Steerer, is_refusal, wilson, repetition, paired_boot
torch.set_grad_enabled(False)
path, tag, part = sys.argv[1], sys.argv[2], sys.argv[3]
R = "/home/amit/.cache/vidyalai-scratch/p09"
S = json.load(open("/home/amit/.cache/vidyalai-scratch/interp/data/p09/splits.json"))
sel = json.load(open(f"{R}/runs/{tag}_select.json"))["selected"]
r = torch.load(f"{R}/runs/{tag}_dirs.pt")["r"][sel["layer"], sel["pos"] + 5]
L = sel["layer"]
rand = torch.randn(r.shape, generator=torch.Generator().manual_seed(0)); rand = rand / rand.norm() * r.norm()
t0 = time.time()
s = Steerer(path)

# FineWeb-Edu held-out documents (Part 2's held-out set), 256 tokens each with <bos>
if "gemma" in tag:
    docs = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")["eval"].long()
else:
    txt = [json.loads(l)["text"] for l in open("/home/amit/.cache/vidyalai-scratch/interp/data/p02/fineweb-edu-heldout600.jsonl")]
    rows = [s.tok(t, add_special_tokens=False).input_ids for t in txt]
    bos = s.tok.bos_token_id if s.tok.bos_token_id is not None else s.tok.convert_tokens_to_ids("<|endoftext|>")
    docs = torch.tensor([[bos] + x[:255] for x in rows if len(x) >= 255])
N_DOC = 200 if part == "static" else 100
docs = docs[:N_DOC]

# ARC-Easy test, 500 random questions (CC BY-SA 4.0)
arc = pq.read_table("/home/amit/.cache/vidyalai-scratch/hf/hub/datasets--allenai--ai2_arc/snapshots/210d026faf9955653af8916fad021475a3f00453/ARC-Easy/test-00000-of-00001.parquet").to_pylist()
random.Random(0).shuffle(arc); arc = arc[:500]
LET = "ABCDE"
def arc_prompt(q):
    opts = "\n".join(f"{LET[i]}. {t}" for i, t in enumerate(q["choices"]["text"]))
    return f"{q['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."
arc_P = s.chat([arc_prompt(q) for q in arc])
arc_gold = [q["choices"]["label"].index(q["answerKey"]) for q in arc]
arc_n = [len(q["choices"]["text"]) for q in arc]
let_ids = [s.tok.encode(c, add_special_tokens=False)[0] for c in LET]

def arc_eval():
    p = s.next_token_probs(arc_P, batch=50)
    correct, is_letter = [], []
    for i in range(len(arc)):
        ids = let_ids[:arc_n[i]]
        correct.append(int(int(p[i, ids].argmax()) == arc_gold[i]))
        is_letter.append(int(int(p[i].argmax()) in ids))
    return np.array(correct), np.array(is_letter)

out = {"model": path, "selected": sel, "n_docs": len(docs), "doc_tokens": docs.shape[1], "n_arc": len(arc)}
if part == "static":
    conds = [("baseline", None), ("add", lambda m: m.add(L, r)), ("add_random", lambda m: m.add(L, rand))]
    if not s.is_gemma:                       # no ablation or subtraction on Gemma (Prohibited Use Policy)
        conds += [("ablate", lambda m: m.ablate(r)), ("subtract", lambda m: m.add(L, -r)),
                  ("ablate_random", lambda m: m.ablate(rand))]
    res = {}
    for name, fn in conds:
        with s.edits(fn):
            ce = s.loss(docs)
            acc, let = arc_eval()
        res[name] = dict(ce=ce.tolist(), arc_correct=acc.tolist(), arc_letter=let.tolist())
        print(name, "CE", ce.mean(), "ARC", acc.mean(), "letter-first", let.mean(), f"{time.time()-t0:.0f}s", flush=True)
    summ = {}
    for name in res:
        ce, acc = np.array(res[name]["ce"]), np.array(res[name]["arc_correct"])
        summ[name] = dict(ce=float(ce.mean()), ppl=float(np.exp(ce.mean())),
                          d_ce=paired_boot(res["baseline"]["ce"], ce),
                          arc_acc=wilson(int(acc.sum()), len(acc)),
                          d_arc=paired_boot(res["baseline"]["arc_correct"], acc),
                          arc_letter_first=float(np.mean(res[name]["arc_letter"])))
    out["summary"] = summ
    json.dump(out, open(f"{R}/post/{tag}_side_static.json", "w"), indent=1)
    json.dump(res, open(f"{R}/runs/{tag}_side_static_raw.json", "w"))
else:
    alphas = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    if s.is_gemma:                           # adding only on Gemma (Prohibited Use Policy)
        alphas = [a for a in alphas if a >= 0]
    hl, hf = s.chat(S["harmless_test"]), s.chat(S["harmful_test"])
    base_ce = None
    rows, texts = [], {}
    for a in alphas:
        with s.edits(None if a == 0 else (lambda m: m.add(L, a * r))):
            ce = s.loss(docs)
            g1 = s.generate(hl, max_new_tokens=64, batch=50)
            g2 = s.generate(hf, max_new_tokens=64, batch=50)
        if a == 0:
            base_ce = ce
        texts[str(a)] = {"harmless": g1, "harmful": g2}
        row = dict(alpha=a, ce=float(ce.mean()), d_ce=paired_boot(base_ce if base_ce is not None else ce, ce) if base_ce is not None else None,
                   harmless_refusal=wilson(sum(map(is_refusal, g1)), len(g1)),
                   harmful_refusal=wilson(sum(map(is_refusal, g2)), len(g2)),
                   harmless_repetition=float(np.mean([repetition(x) for x in g1])),
                   harmless_ce_list=ce.tolist())
        rows.append(row)
        print(a, "CE", ce.mean(), "refuse harmless", row["harmless_refusal"][0], "harmful", row["harmful_refusal"][0],
              "rep", row["harmless_repetition"], f"{time.time()-t0:.0f}s", flush=True)
    # recompute paired deltas against alpha=0 for the negative alphas run before it
    b = np.array(next(x for x in rows if x["alpha"] == 0)["harmless_ce_list"])
    for x in rows:
        x["d_ce"] = paired_boot(b, x["harmless_ce_list"]); del x["harmless_ce_list"]
    out["sweep"] = rows
    json.dump(out, open(f"{R}/post/{tag}_sweep.json", "w"), indent=1)
    json.dump(texts, open(f"{R}/runs/{tag}_sweep_texts.json", "w"))
print("done", f"{time.time()-t0:.0f}s")
