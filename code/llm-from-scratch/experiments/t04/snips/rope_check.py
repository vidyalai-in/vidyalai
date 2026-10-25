import torch
from model import MAIN, rope_tables, apply_rope

torch.manual_seed(0)
cos, sin = rope_tables(MAIN)                        # two (1024, 32) tables
q, k = torch.randn(64), torch.randn(64)
Q = apply_rope(q.expand(1, 1, 1024, 64), cos, sin)[0, 0]   # q placed at every position
K = apply_rope(k.expand(1, 1, 1024, 64), cos, sin)[0, 0]
for m, n in [(5, 3), (105, 103), (1005, 1003), (5, 4)]:
    print(f"query at {m:4}, key at {n:4}: score {(Q[m] @ K[n]).item():.4f}")
print(f"length before {q.norm():.4f}, after {Q[700].norm():.4f}")
