import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/v04"); sys.argv=["x","none"]
import claude_run as C
for it in [i for i in C.items if i.task in ("count","read")][:2] + [i for i in C.items if i.task=="read"][:3]:
    for size in (224, 896):
        p = C.params(it, size); p["max_tokens"] = 400
        msg = C.ledger.create(C.cl, "v04", note=f"v04 pilot2 {it.id} {size}", **p)
        text = "".join(c.text for c in msg.content if c.type == "text")
        r=dict(id=it.id,size=size,answer=it.answer,text=text,stop=msg.stop_reason,out=msg.usage.output_tokens)
        open("claude_pilot.jsonl","a").write(json.dumps(r)+"\n"); print(json.dumps(r))
