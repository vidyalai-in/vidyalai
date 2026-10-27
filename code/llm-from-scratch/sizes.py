"""Model sizes for the scaling sweep (Build an LLM From Scratch, Part 8).

The same family as SMALL and MAIN: grouped-query attention with three query heads per
key/value head, a SwiGLU width of exactly 8/3 d, the 32,768-token vocabulary, 1,024 context.
Width grows in steps the 3:1 head layout allows; head_dim is 64 where the width permits
(384, 576, 768), else 80 or 48. Depth grows slowly, so d/n_layer rises from 48 to 77.

    python sizes.py        # the table: total and non-embedding parameters per size
"""
from model import GPTConfig, MAIN, SMALL, param_formula

S11 = GPTConfig(d_model=240, n_layer=5, n_head=3, n_kv_head=1, d_ff=640)    # 10,938,960 parameters
S15 = GPTConfig(d_model=288, n_layer=6, n_head=6, n_kv_head=2, d_ff=768)    # 14,749,344
S33 = GPTConfig(d_model=480, n_layer=7, n_head=6, n_kv_head=2, d_ff=1280)   # 32,939,040
S47 = GPTConfig(d_model=576, n_layer=8, n_head=9, n_kv_head=3, d_ff=1536)   # 47,195,712

SIZES = {"s11": S11, "s15": S15, "s22": SMALL, "s33": S33, "s47": S47}


def counts(cfg):
    """(total, non-embedding) parameters; the tied embedding table is counted once, in total only."""
    parts = param_formula(cfg)
    total = sum(parts.values())
    return total, total - parts["embedding"]


if __name__ == "__main__":
    print(f"{'size':5} {'d':>4} {'L':>3} {'heads':>6} {'hd':>3} {'d_ff':>5} {'total':>11} {'non-emb':>11} {'emb share':>9}")
    for name, cfg in {**SIZES, "main": MAIN}.items():
        total, ne = counts(cfg)
        print(f"{name:5} {cfg.d_model:4} {cfg.n_layer:3} {cfg.n_head:>3}/{cfg.n_kv_head:<2} {cfg.head_dim:3} "
              f"{cfg.d_ff:5} {total:11,} {ne:11,} {1 - ne / total:9.1%}")
