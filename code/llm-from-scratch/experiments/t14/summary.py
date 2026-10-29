"""One line per bench.py run: memory, speed, clocks (from nvidia-smi logs); writes t14/post/runs_summary.json."""
import glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
R = "/home/amit/.cache/vidyalai-scratch/t14/runs"; P = "/home/amit/.cache/vidyalai-scratch/t14/post"
os.system(f"python3 {os.path.dirname(__file__)}/clocks.py > /dev/null")
clk = json.load(open(f"{P}/clocks.json"))
G = 2 ** 30
out = {}
for f in sorted(glob.glob(f"{R}/*.json")):
    t = os.path.basename(f)[:-5]
    if t.startswith("precision"):
        continue
    r = json.load(open(f))
    c = clk.get(t, {})
    row = {"params": r.get("params"), "trainable": r.get("trainable"), "oom": r["oom"], "mb": r["args"]["mb"],
           "peak_alloc_gib": round(r["peak_alloc"] / G, 3), "tok_per_s": r.get("tok_per_s"),
           "peak_reserved_gib": round(r["peak_reserved"] / G, 3) if r.get("peak_reserved") else None,
           "sm_med": c.get("sm_med"), "power_med": c.get("power_med"), "limit": c.get("limit"),
           "opt_s": [round(x, 3) for x in r.get("opt_s", [])], "step_s": [round(x, 2) for x in r.get("step_s", [])],
           "host_maxrss_gib": round(r["host_maxrss"] / G, 2) if r.get("host_maxrss") else None,
           "error": r.get("error", "")[:90]}
    if row["tok_per_s"] and row["sm_med"]:
        row["tok_per_s_per_ghz"] = round(row["tok_per_s"] / row["sm_med"] * 1000)
    out[t] = row
json.dump(out, open(f"{P}/runs_summary.json", "w"), indent=1)
for t, r in out.items():
    if len(sys.argv) > 1 and not any(t.startswith(p) for p in sys.argv[1:]):
        continue
    print(f"{t:22} P={r['params']/1e6 if r['params'] else 0:7.1f}M mb={r['mb']:3} oom={str(r['oom'])[0]} alloc={r['peak_alloc_gib']:5.2f} res={r['peak_reserved_gib']} tok/s={r['tok_per_s']} perGHz={r.get('tok_per_s_per_ghz')} sm={r['sm_med']} opt={r['opt_s']} rss={r['host_maxrss_gib']} {r['error'][:60]}")
