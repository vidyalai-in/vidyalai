"""Pilot 4: variable tracking with room to reason (max_tokens 256), graded on the last line."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc
paras = lc.load_corpus()
cl = ledger.client()
OUT = "/home/amit/.cache/vidyalai-scratch/e09/runs/pilot-vt.jsonl"
for n_chars in [0, int((16000 - 1050) * 3.07)]:
    hay = lc.build_haystack(paras, n_chars, 500, 101)
    for qid, task, q, gold, depth in lc.questions(hay)[12:]:
        p = lc.claude_params(hay["text"], q, ttl="5m", max_tokens=256)
        r = ledger.create(cl, "e09", note=f"pilot-vt {n_chars} {qid}", **p)
        reply = "".join(c.text for c in r.content if c.type == "text")
        with open(OUT, "a") as f:
            f.write(json.dumps(dict(n_chars=n_chars, qid=qid, gold=gold, reply=reply, stop=r.stop_reason, usage=r.usage.model_dump())) + "\n")
        print(n_chars, qid, lc.grade_vt(reply, gold), r.stop_reason, r.usage.output_tokens, repr(reply[-160:]), flush=True)
