"""plan.py: steps, tokens and predicted hours per run, from sizes.py and measured tokens/s."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from sizes import SIZES, counts
# tokens per step, measured tok/s at that batch (micro-batch 16, compiled, fused)
B = {"s11": 65536, "s15": 65536, "s22": 131072, "s33": 262144, "s47": 262144}
TPS = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])}
C15 = 6 * counts(SIZES["s15"])[0] * 20 * counts(SIZES["s15"])[0]
rows = [(n, n, 20) for n in B] + [("iso_s11", "s11", None), ("iso_s22", "s22", None), ("iso_s33", "s33", None)]
tot = 0
print(f"{'run':8} {'total N':>11} {'non-emb':>11} {'tok/param':>9} {'tokens':>13} {'tok/step':>8} {'steps':>6} {'warmup':>6} {'val':>4} {'ckpt':>4} {'tok/s':>7} {'hours':>5}")
for run, m, tpp in rows:
    N, ne = counts(SIZES[m])
    D = 20 * N if tpp else C15 / (6 * N)
    steps = round(D / B[m]); tokens = steps * B[m]
    h = tokens / TPS[m] / 3600; tot += h
    print(f"{run:8} {N:11,} {ne:11,} {tokens / N:9.2f} {tokens:13,} {B[m]:8,} {steps:6} {round(0.045 * steps):6} "
          f"{round(steps / 40):4} {round(steps / 20):4} {TPS[m]:7,.0f} {h:5.2f}  cum {tot:5.2f}")
