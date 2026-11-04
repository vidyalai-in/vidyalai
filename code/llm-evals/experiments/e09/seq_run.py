"""Sequential cached calls on one haystack: the first request writes the cache, the rest read it.
usage: seq_run.py <run_name> <target_tokens> <start_para> <seed> [qids comma list|all] [ttl 5m|1h]"""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc
name, target, start, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
want = sys.argv[5] if len(sys.argv) > 5 else "all"
ttl = sys.argv[6] if len(sys.argv) > 6 else "5m"
OUT = f"/home/amit/.cache/vidyalai-scratch/e09/runs/{name}.jsonl"
paras = lc.load_corpus()
n_chars = int((target - 1050) * 3.07)
hay = lc.build_haystack(paras, n_chars, start, seed)
qs = lc.questions(hay)
if want.startswith("idx="):
    qs = [qs[int(i)] for i in want[4:].split(",")]
elif want != "all":
    keep = want.split(",")
    qs = [q for q in qs if q[0] in keep]
cl = ledger.client()
meta = dict(event="haystack", name=name, target=target, n_chars=len(hay["text"]), start_para=start, seed=seed,
            chars=hay["chars"], chains=hay["chains"])
with open(OUT, "a") as f:
    f.write(json.dumps(meta) + "\n")
for qid, task, qtext, gold, depth in qs:
    p = lc.claude_params(hay["text"], qtext, ttl=ttl, max_tokens={"vt": 256}.get(task, 48))
    t0 = time.time()
    r = ledger.create(cl, "e09", note=f"{name} {qid}", **p)
    dt = time.time() - t0
    reply = "".join(c.text for c in r.content if c.type == "text")
    u = r.usage.model_dump()
    cost = ledger.cost_of("claude-haiku-5-5", u)
    rec = dict(event="answer", name=name, qid=qid, task=task, question=qtext, gold=gold, depth=depth, reply=reply,
               stop=r.stop_reason, usage=u, cost=cost, seconds=round(dt, 2), batch=False)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(qid, task, repr(reply[:80]), r.stop_reason, u["input_tokens"], u.get("cache_creation_input_tokens"),
          u.get("cache_read_input_tokens"), u["output_tokens"], f"${cost:.5f}", f"{dt:.1f}s", flush=True)
