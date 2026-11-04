"""Grade every MATH-500 reply three ways (exact, normalized, symbolic) from the same extracted \\boxed{}."""
import json, sys, glob
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify
from math_verify import parse, verify as mv
D = "/home/amit/.cache/vidyalai-scratch/evals/data/"; R = "/home/amit/.cache/vidyalai-scratch/e08/runs/"
m5 = json.load(open(D + "e08-math500.json"))
rows = []
for path in sorted(glob.glob(R + "math_haiku_*_r*.jsonl")):
    for l in open(path):
        r = json.loads(l)
        setting, rep, i = r["id"].split("-"); i = int(i)
        gold = m5[i]["answer"]; text = r.get("text") or ""
        boxed = verify.last_boxed(text)
        pred, rule = verify.extract_answer(text)
        row = dict(id=r["id"], setting=setting, rep=rep, item=i, subject=m5[i]["subject"], level=m5[i]["level"],
                   gold=gold, pred=pred, rule=rule, boxed=boxed, stop=r.get("stop"), out_tokens=(r.get("usage") or {}).get("output_tokens"),
                   thinking="thinking" in (r.get("types") or []))
        for g, f in verify.GRADERS.items():
            row[g] = bool(f(pred, gold))
            row[g + "_boxonly"] = bool(f(boxed, gold))
        try:  # math-verify with its own answer extraction on the whole reply
            row["symbolic_full"] = bool(mv(parse("$" + gold + "$"), parse(text)))
        except Exception:
            row["symbolic_full"] = False
        rows.append(row)
json.dump(rows, open(R + "math_graded.json", "w"))
import collections
c = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    for g in ("exact", "normalized", "symbolic", "symbolic_full"):
        c[(r["setting"], r["rep"])][g] += r[g]
    c[(r["setting"], r["rep"])]["n"] += 1; c[(r["setting"], r["rep"])]["nobox"] += r["boxed"] is None
    c[(r["setting"], r["rep"])]["noans"] += r["pred"] is None
    c[(r["setting"], r["rep"])]["trunc"] += r["stop"] == "max_tokens"
for k, v in sorted(c.items()):
    print(k, dict(v))
