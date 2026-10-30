from lab import dynamics as D

for step in (300, 600, 3360):
    model = D.load(f"{D.RUNS['main88'].path}/model_{step:05d}.pt", D.RUNS["main88"].cfg)  # FP32
    tok = D.repeated_tokens(200, 50, seed=1, allowed=D.frequent_tokens())              # [EOT] + r + r
    s = D.head_scores(model, tok, half=50)              # Part 4's scores for all 120 heads
    ind = s["induction"].mean(0)                         # [layer, head]
    l, h = divmod(int(ind.argmax()), ind.shape[1])
    print(f"step {step}: pattern check {D.check_patterns(model, tok[:8]):.0e}, "
          f"best induction head {l}.{h} ({ind.max():.2f}), "
          f"loss first copy {s['loss'][:, 1:50].mean():.1f}, second copy {s['loss'][:, 51:].mean():.1f}")
