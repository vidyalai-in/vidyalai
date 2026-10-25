import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
RUNS = "/home/amit/.cache/vidyalai-scratch/a02/runs"

def dump(path, rec):
    with open(f"{RUNS}/{path}", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")

def ci95(k, n):
    """Wilson score interval for k successes in n trials."""
    import math
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c - h), min(1, c + h))
