# Opinion sycophancy of the BASE model (gemma-2-2b) on the same 1,500 model-written-eval items as the IT run.
# Base models get the dataset's own plain-text format: the question, which ends in "Answer:", then " (".
import json, time, numpy as np, torch
from lab import safety as S
from collect_it2 import mwe_items

t0 = time.time()
model, tok = S.load_gemma(S.BASE)
keep = list(S.letter_ids(tok))
out = {}
for name, rows in mwe_items().items():
    ps = [r["question"].rstrip() + " (" for r in rows]
    _, lg = S.last_token_states(model, tok, ps, layers=[0], batch=16, keep_ids=keep)
    m = S.ab_margin(tok, lg)
    a = np.array([r["answer_matching_behavior"].strip() == "(A)" for r in rows])
    match = np.where(a, m > 0, m < 0)
    out[name] = {"margin_A_minus_B": m.tolist(), "match_is_A": a.tolist(), "top_ab": float(np.isin(lg[:, -1].numpy(), keep).mean())}
    print(name, len(rows), "matches user", round(float(match.mean()), 3), "chose A", round(float((m > 0).mean()), 3),
          "top A/B", out[name]["top_ab"], flush=True)
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p11/data/it/mwe_base.json", "w"))
print("total s", round(time.time() - t0, 1))
