import json, sys, statistics as st
sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/a09')
from analyze import summary, fmt
model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
rows=[json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/a09/results.jsonl")]
rows=[r for r in rows if r["model"]==model and r.get("label","")==""]
for arm in ["computer","text","hybrid"]:
    rs=[r for r in rows if r["arm"]==arm]
    if not rs: continue
    print(arm, fmt(summary(rs)))
    for k in ("info","action"):
        print("  ",k, fmt(summary([r for r in rs if r["kind"]==k])))
    print("   median wall", st.median(r["wall"] for r in rs), "median model", st.median(r["latency"] for r in rs), "median cost", round(st.median(r["cost"] for r in rs),5), "max steps", max(r["steps"] for r in rs), "stops", {s:sum(r['stop']==s for r in rs) for s in set(r['stop'] for r in rs)})
    print("   failures:", [(r["task"], r["rep"], r["answer"][:60]) for r in rs if not r["ok"]])
