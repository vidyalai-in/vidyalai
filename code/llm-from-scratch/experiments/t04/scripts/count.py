import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from model import GPT, MAIN, SMALL, param_formula

for name, cfg in [("main", MAIN), ("small", SMALL)]:
    parts = param_formula(cfg)
    model = GPT(cfg)
    print(name, {k: f"{v:,}" for k, v in parts.items()})
    print(f"  formula {sum(parts.values()):,}   code {model.num_params():,}")
