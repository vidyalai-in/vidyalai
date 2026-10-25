"""Startup latency and per-call overhead of each sandbox backend (no API calls)."""
import json, statistics, subprocess, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import sandbox as sb

N = 30
ARGV = ["python3", "-c", "pass"]
work = sb.new_workdir("bench")
out = {}

def plain():
    return subprocess.run(ARGV, cwd=work, capture_output=True, text=True, timeout=30,
                          env={"PATH": "/usr/bin:/bin"})

def record(label, fn):
    fn()  # warm-up, not counted
    ts = []
    for _ in range(N):
        t, r = sb.timed(fn)
        assert r.returncode == 0, (label, r.stderr[:300])
        ts.append(t * 1000)
    ts.sort()
    out[label] = {"n": N, "median_ms": round(statistics.median(ts), 1),
                  "p95_ms": round(ts[int(0.95 * N) - 1], 1), "min_ms": round(ts[0], 1)}
    print(label, out[label], flush=True)

record("plain subprocess", plain)
record("bwrap", lambda: sb.run_bwrap(work, ARGV))
record("docker run (fresh)", lambda: sb.run_docker_fresh(work, ARGV, f"{sb.PREFIX}-bench"))
t, warm = sb.timed(sb.WarmContainer, work, f"{sb.PREFIX}-warm")
out["warm container start_ms"] = round(t * 1000, 1)
try:
    record("docker exec (warm)", lambda: warm.run(ARGV))
finally:
    warm.close()
json.dump(out, open(sb.SCRATCH / "runs" / "bench.json", "w"), indent=1)
