# Build chart data blocks from post/summary.json and write the repo's diagram generators.
import json, math, sys
S = json.load(open("/home/amit/.cache/vidyalai-scratch/p10/post/summary.json"))
REPO = "/home/amit/AlpsCode/ai/vidyalai/diagrams"
T = "/home/amit/.cache/vidyalai-scratch/p10/scripts/charts"
def emit(name, data):
    src = open(f"{T}/{name}.tpl.py").read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(f"{REPO}/{name}.py", "w").write(src)
m = S["runs"]["main88"]
val = sorted((int(k), v) for k, v in S["runs_val_main88"].items())
slope = [[math.sqrt(a[0] * b[0]), round(-(b[1] - a[1]) / math.log(b[0] / a[0]), 4)] for a, b in zip(val, val[1:]) if a[0] >= 50]
emit("p10-phase-change", {
    "slope": slope,
    "steps": [r["step"] for r in m if r["step"] > 0],
    "ind": [r["ind_max"] for r in m if r["step"] > 0],
    "prev": [r["prev_low_max"] for r in m if r["step"] > 0],
    "dup": [r["dup_max"] for r in m if r["step"] > 0],
    "second": [r["second"][0] for r in m if r["step"] > 0],
    "first": [r["first"][0] for r in m if r["step"] > 0],
    "icl": [r["icl_s"] for r in m if r["step"] > 0]})
if "--ngram" in sys.argv or True:
    emit("p10-ngram", {"steps": [r["step"] for r in m if r["step"] > 0],
                       "kl_uni": [r["kl_uni"][0] for r in m if r["step"] > 0], "kl_bi": [r["kl_bi"][0] for r in m if r["step"] > 0],
                       "uni": [r["uni_loss"][0] for r in m if r["step"] > 0], "bi": [r["bi_loss"][0] for r in m if r["step"] > 0],
                       "H_uni": S["H_uni"], "H_bi": S["H_bi"]})
print("ok")
pr = sorted(S["probe"], key=lambda r: r["step"])
emit("p10-pos-probe", {"steps": [r["step"] for r in pr], "lookup": 0.8405,
    "l8": [r["layers"]["8"]["acc"][0] for r in pr], "l8u": [r["layers"]["8"]["unseen"] for r in pr],
    "l0": [r["layers"]["0"]["acc"][0] for r in pr], "l8c": [r["layers"]["8"]["control"] for r in pr]})
NAMES = [("s11", "11M (Part 8)"), ("s15", "15M (Part 8)"), ("s22", "22M (Part 8)"), ("s33", "33M (Part 8)"), ("s47", "47M (Part 8)"), ("main88", "88M (Part 7)")]
runs = []
for k, nm in NAMES:
    rows = [r for r in S["runs"][k] if r["step"] > 0]
    runs.append({"name": nm, "steps": [r["step"] for r in rows], "tokens": [r["tokens"] for r in rows], "ind": [r["ind_max"] for r in rows]})
py = S.get("pythia70m", {})
ks = sorted(py, key=int)
runs.append({"name": "Pythia-70m (public)", "steps": [int(k) for k in ks], "tokens": [py[k]["tokens"] for k in ks], "ind": [max(max(r) for r in py[k]["induction"]) for k in ks]})
emit("p10-sizes", {"runs": runs})
print("ok2")
