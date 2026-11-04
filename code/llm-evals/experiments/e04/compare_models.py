"""Haiku 5.5 vs Sonnet 5.5 on the same 285 items: the gap under each extractor and prompt, and rotation consistency."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq
def load(tag):
    by = {}
    for l in open(f"runs/batch-{tag}.jsonl"):
        r = json.loads(l); by.setdefault(r["cond"], {})[r["i"]] = r
    return by
H, S = load("claude-haiku-5-5"), load("sonnet-285")
idx = sorted(S["mmlu"])
ITEMS = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
subj = [ITEMS[i]["subject"] for i in idx]
ok = lambda by, c, i, ex: int(mcq.EXTRACTORS[ex](by[c][i]["text"], {"choices": by[c][i]["choices"]}) == by[c][i]["answer"])
out = {}
for c in ("raw", "mmlu"):
    for ex in mcq.EXTRACTORS:
        h = [ok(H, c, i, ex) for i in idx]; s = [ok(S, c, i, ex) for i in idx]
        p = mcq.paired(s, h, subj)
        out[f"{c}:{ex}"] = dict(haiku=sum(h)/len(h), sonnet=sum(s)/len(s), **p)
        print(f"{c:5s} {ex:9s} haiku {100*sum(h)/len(h):5.1f} sonnet {100*sum(s)/len(s):5.1f}  gap {100*p['diff']:+5.1f} [{100*p['low']:+.1f}, {100*p['high']:+.1f}]")
rot = ["mmlu", "rot1", "rot2", "rot3"]
for name, by in (("haiku", H), ("sonnet", S)):
    same = allr = 0
    for i in idx:
        ps = [mcq.x_flex(by[c][i]["text"], {"choices": by[c][i]["choices"]}) for c in rot]
        back = [None if p is None else (p - r) % 4 for r, p in enumerate(ps)]
        same += len(set(back)) == 1 and back[0] is not None
        allr += all(p == by[c][i]["answer"] for p, c in zip(ps, rot))
    mean = sum(ok(by, c, i, "flex") for c in rot for i in idx) / (4 * len(idx))
    out[f"{name}_rot"] = dict(mean=mean, consistent=mcq.wilson(same, len(idx)), all4=mcq.wilson(allr, len(idx)))
    print(name, "mean over rotations", round(100*mean, 1), "consistent", round(100*same/len(idx), 1), "all four right", round(100*allr/len(idx), 1))
json.dump(out, open("runs/compare-haiku-sonnet-285.json", "w"), indent=1)
