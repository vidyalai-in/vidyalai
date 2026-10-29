import json, math, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"; REPO = "/home/amit/AlpsCode/ai/vidyalai/diagrams/"
def write(tpl, name, data):
    s = open(f"/home/amit/.cache/vidyalai-scratch/p06/scripts/{tpl}").read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(REPO + name + ".py", "w").write(s)
med = lambda v: sorted(v)[len(v) // 2]
# 1 sweep
r = json.load(open(RUN + "sweep.json")); rl = json.load(open(RUN + "sweep_linear.json")); R = r["reps"]
def tri(key, rr):
    rows = [rr[key][k*R:(k+1)*R] for k in range(len(rr["densities"]))]
    return [[min(x) for x in rows], [med(x) for x in rows], [max(x) for x in rows]]
write("tpl_sweep.py", "p06-features-sparsity", dict(dens=r["densities"], rep=tri("learned", r), sup=tri("superposed", r), lin=tri("learned", rl)))
# 2 phase
write("tpl_phase.py", "p06-phase-diagram", dict(p2=json.load(open(RUN + "phase2.json"))["best"], p3=json.load(open(RUN + "phase3.json"))["best"]))
# 3 geometry
g = json.load(open(RUN + "geometry.json")); R = g["reps"]
pts = set()
for k, d in enumerate(g["densities"]):
    for row in g["dim"][k*R:(k+1)*R]:
        for v in row:
            pts.add((round(1 / d, 3), round(v / 0.005) * 0.005))
pts = sorted(pts)
ds = [(round(1 / d, 3), sum(g["dstar"][k*R:(k+1)*R]) / R) for k, d in enumerate(g["densities"])]
write("tpl_geom.py", "p06-dimensionality", dict(pts=[[x, round(y, 3)] for x, y in pts], dstar=[[x, round(y, 4)] for x, y in ds]))
print("geometry points", len(pts))
# 4 pentagon
p = json.load(open(RUN + "pentagon.json")); R = p["reps"]
pan = []
for d in (0.5, 0.2, 0.05, 0.01):
    k = p["densities"].index(d)
    loss = p["loss"][k*R:(k+1)*R]; b = min(range(R), key=lambda i: loss[i])
    dims = p["dim"][k*R:(k+1)*R]
    pent = sum(1 for row in dims if all(0.37 < v < 0.43 for v in row))
    pairs = sum(1 for row in dims if sorted(round(v, 1) for v in row) == [0.0, 0.5, 0.5, 0.5, 0.5])
    lines = [f"pentagon: {pent} of 50 runs", f"two pairs: {pairs} of 50 runs"]
    pan.append(dict(title=f"Density {d:g} (1 in {round(1/d)})", W=[[round(a, 3), round(c, 3)] for a, c in p["W"][k*R+b]], lines=lines))
write("tpl_pent.py", "p06-pentagon", pan)
# 5 gpt2
a = json.load(open(RUN + "s7_analysis.json")); s6 = json.load(open(RUN + "s6_analysis.json"))
write("tpl_gpt2.py", "p06-gpt2-neurons", dict(multi=[l["multi"] for l in a["layers"]], mixed=[l["mixed"] for l in a["layers"]],
      tokspec=[l["tokspec"] for l in a["layers"]], multi_ci=s6["boot"]["layer_2plus"]))
