from model import GPT, SMALL, MAIN
for name, cfg in [("small", SMALL), ("main", MAIN)]:
    model = GPT(cfg)
    decay = [n for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [n for n, p in model.named_parameters() if p.dim() < 2]
    count = lambda names: sum(dict(model.named_parameters())[n].numel() for n in names)
    print(f"{name}: decay {len(decay)} tensors, {count(decay):,} params; "
          f"no decay {len(no_decay)} tensors, {count(no_decay):,} params")
