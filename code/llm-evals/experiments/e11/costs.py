import json, collections
c = collections.defaultdict(float); n = collections.Counter()
for l in open("/home/amit/.cache/vidyalai-scratch/agents/lab/ledger.jsonl"):
    e = json.loads(l)
    if e["part"] != "e11": continue
    k = e["note"].split()[0] if not e["note"].startswith("prewarm") else "prewarm"
    if e["note"].startswith("draft"): k = "draft"
    c[k] += e["cost"]; n[k] += 1
for k in c: print(f"{k:10s} {n[k]:5d} ${c[k]:.4f}")
print("total", f"${sum(c.values()):.4f}")
