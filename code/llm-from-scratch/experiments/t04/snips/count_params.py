from model import GPT, MAIN, SMALL, param_formula

for name, cfg in [("main", MAIN), ("small", SMALL)]:
    parts = param_formula(cfg)
    print(name, {k: f"{v:,}" for k, v in parts.items()})
    print(f"  formula {sum(parts.values()):,}   code {GPT(cfg).num_params():,}")
