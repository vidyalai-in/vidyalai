"""Pilot 3: NoLiMa's template, at 0 filler (needles only) and on the 16K pilot haystack."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc
paras = lc.load_corpus()
cl = ledger.client()
OUT = "/home/amit/.cache/vidyalai-scratch/e09/runs/pilot-wording2.jsonl"
for n_chars in [0, int((16000 - 1050) * 3.07)]:
    hay = lc.build_haystack(paras, n_chars, 500, 101)
    qs = lc.questions(hay)
    for qid, task, q, gold, depth in [qs[1], qs[3], qs[5], qs[12], qs[13]]:
        p = lc.claude_params(hay["text"], q, ttl="5m", max_tokens=64)
        r = ledger.create(cl, "e09", note=f"pilot-wording2 {n_chars} {qid}", **p)
        reply = "".join(c.text for c in r.content if c.type == "text")
        g = lc.grade_name(reply, gold) if task != "vt" else lc.grade_vt(reply, gold)
        with open(OUT, "a") as f:
            f.write(json.dumps(dict(n_chars=n_chars, qid=qid, task=task, gold=gold, reply=reply, stop=r.stop_reason, usage=r.usage.model_dump())) + "\n")
        print(n_chars, qid, g, repr(reply[:120]), r.stop_reason, flush=True)
