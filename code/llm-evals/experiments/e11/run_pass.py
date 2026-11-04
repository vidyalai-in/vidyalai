"""Run one pass of the eval through the Message Batches API.
usage: run_pass.py <config: off|low|medium> <tag> [closed] [--parts 1,2,...] [--prewarm]
Items: data/items.jsonl (validated) or data/candidates.jsonl when tag starts with 'val'."""
import sys, json, time, argparse
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, own_eval as oe
ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("tag"); ap.add_argument("closed", nargs="?", default="")
ap.add_argument("--parts", default="1,2,3,4,5,6,7,8"); ap.add_argument("--prewarm", action="store_true"); ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--items", default=None); ap.add_argument("--reps", type=int, default=1)
a = ap.parse_args()
closed = a.closed == "closed"
src = a.items or ("data/candidates.jsonl" if a.tag.startswith("val") else "data/items.jsonl")
items = [json.loads(l) for l in open(src)]
parts = [int(x) for x in a.parts.split(",")]
items = [it for it in items if it["part"] in parts and not (closed and it["kind"] == "none")]
arts = oe.load_articles()
tok = json.load(open("data/article_tokens.json"))
cl = ledger.client()
if a.prewarm and not closed:
    for n in parts:  # write each article's prefix to the 1-hour cache with one cheap call
        p = oe.answer_request({"question": "Warm-up: reply with the word ok."}, arts[n], a.config); p["max_tokens"] = 16
        r = ledger.create(cl, "e11", note=f"prewarm part {n}", **p)
        print("prewarm", n, r.usage.cache_creation_input_tokens, r.usage.cache_read_input_tokens, flush=True)
groups = [parts[i::a.groups] for i in range(a.groups)]
out = f"runs/{a.tag}.jsonl"
total = 0.0
for g in groups:
    jobs, bound = {}, {}
    for it in items:
        if it["part"] not in g: continue
        p = oe.answer_request(it, arts[it["part"]], a.config, closed=closed)
        for r in range(a.reps):
            cid = f"r{r+1}_{it['id']}"
            jobs[cid] = p
            bound[cid] = (300 if closed else tok[str(it["part"])] + 600) + len(it["question"]) // 2
    t0 = time.time()
    bid, est = oe.submit("e11", jobs, bound, "runs/batch_ids.jsonl")
    print("submitted", bid, len(jobs), f"worst ${est:.4f}", flush=True)
    c = oe.collect("e11", bid, jobs, out, note=f"{a.tag} {a.config}{' closed' if closed else ''}")
    total += c
    print("done", bid, f"${c:.4f}", f"{time.time()-t0:.0f}s", flush=True)
print("TOTAL", f"${total:.4f}")
