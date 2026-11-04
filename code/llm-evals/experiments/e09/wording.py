"""Pilot 2: question wordings for the one-hop and variable-tracking tasks on the pilot haystack."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc
paras = lc.load_corpus()
hay = lc.build_haystack(paras, int((16000 - 1050) * 3.07), 500, 101)
cl = ledger.client()
W = {
 "home": "Which character's home is in {city}? Answer with only the character's name.",
 "beento_infer": "Which character has been to {city}? The text may not say so directly. Answer with only the character's name.",
}
VT = {
 "ruler": "Find all variables that are assigned the value {value} in the text above, directly or through a chain of assignments (for example VAR B = VAR A, where VAR A = {value}). List every variable in the chain, separated by commas, and nothing else.",
}
OUT = "/home/amit/.cache/vidyalai-scratch/e09/runs/pilot-wording.jsonl"
jobs = []
for ch in hay["chars"][:4]:
    for k, w in W.items():
        jobs.append((f"{k}-{ch['name']}", w.format(**ch), ch["name"]))
for j, c in enumerate(hay["chains"]):
    for k, w in VT.items():
        jobs.append((f"{k}-vt{j}", w.format(value=c["value"]), c["vars"]))
for qid, q, gold in jobs:
    p = lc.claude_params(hay["text"], q, ttl="5m", max_tokens=96)
    r = ledger.create(cl, "e09", note=f"pilot-wording {qid}", **p)
    reply = "".join(c.text for c in r.content if c.type == "text")
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(qid=qid, question=q, gold=gold, reply=reply, stop=r.stop_reason, usage=r.usage.model_dump())) + "\n")
    print(qid, gold if isinstance(gold, str) else "", repr(reply[:150]), r.stop_reason, r.usage.cache_read_input_tokens, flush=True)
