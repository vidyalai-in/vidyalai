import json, collections
rs=[json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/e07/runs/gen60.jsonl")]
by=collections.defaultdict(list)
for r in rs: by[r["id"].split("-",1)[1]].append(r)
for k,v in by.items():
    out=[r["usage"]["output_tokens"] for r in v]; stops=collections.Counter(r["stop"] for r in v)
    print(k, len(v), "mean out", round(sum(out)/len(out)), stops, "thought", sum(r["thought"] for r in v))
