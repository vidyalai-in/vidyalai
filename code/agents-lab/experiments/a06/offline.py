"""Free measurements: BM25 recall@k over the toolsets, and token counts per arm (count_tokens)."""
import sys
from run import toolset, bm25_rank, build, REQUESTS, CATALOG
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger

SIZES = [5, 20, 50, 100, 145]
print("BM25 recall@k (target among top k), 79 requests")
for n in SIZES:
    ranks = []
    for r in REQUESTS:
        names = [t["name"] for t in bm25_rank(r["text"], toolset(r, n))]
        ranks.append(names.index(r["tool"]) + 1)
    print(n, {k: sum(x <= k for x in ranks) for k in (1, 3, 5, 10)})

if "--tokens" in sys.argv:
    cl = ledger.client()
    r = REQUESTS[0]
    msg = [{"role": "user", "content": r["text"]}]
    for arm in ("all", "search_bm25", "client_bm25"):
        for n in SIZES:
            tools, system = build(arm, r, n, 5)
            c = cl.messages.count_tokens(model="claude-haiku-5-5", system=system, tools=tools, messages=msg)
            print(arm, n, c.input_tokens)
    c = cl.messages.count_tokens(model="claude-haiku-5-5", system=[{"type": "text", "text": "x"}], messages=msg)
    print("no tools, system x", c.input_tokens)
