"""Print the data blocks pasted into diagrams/v04-*.py from results.json."""
import json
r = json.load(open("/home/amit/.cache/vidyalai-scratch/v04/results.json"))
A = r["accuracy"]
M = {"Claude Haiku 5.5": "haiku|thinking off", "Claude Sonnet 5.5": "sonnet|between_tools",
     "Qwen2-VL-2B": "qwen2vl-2b|default", "SmolVLM-256M": "smolvlm-256m|default"}
acc = {}
for name, key in M.items():
    for task, d in A.get(key, {}).items():
        acc.setdefault(task, {})[name] = [(int(s), round(100 * v["p"], 1), round(100 * v["lo"], 1), round(100 * v["hi"], 1)) for s, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
print("ACC =", json.dumps(acc))
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vlm
BINS = [(3, 4), (5, 6.5), (8, 10), (12, 20), (24, 68)]   # font size in the final image, pixels
h = {}
for name, key in M.items():
    d = r["read_by_height"].get(key)
    if not d:
        continue
    pts = []
    for lo_b, hi_b in BINS:
        ks = [v for x, v in d.items() if lo_b <= float(x) <= hi_b]
        if not ks:
            continue
        k, n = sum(v["k"] for v in ks), sum(v["n"] for v in ks)
        p, lo, hi = vlm.wilson(k, n)
        pts.append((round((lo_b * hi_b) ** 0.5, 2), round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1), n))
    h[name] = pts
print("HEIGHT =", json.dumps(h))
