import numpy as np
T = 1000
betas = np.linspace(1e-4, 0.02, T)          # Ho et al.'s linear schedule
alpha_bar = np.cumprod(1 - betas)
rng = np.random.default_rng(0)
x0, t = 0.8, 300                             # one bright pixel, 300 noising steps
x = np.full(100_000, x0)                     # 100,000 copies of it
for b in betas[:t]:                          # the slow way: 300 small steps
    x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.standard_normal(x.shape)
print(f"300 steps:   mean {x.mean():.4f}  std {x.std():.4f}")
ab = alpha_bar[t - 1]
print(f"closed form: mean {np.sqrt(ab) * x0:.4f}  std {np.sqrt(1 - ab):.4f}")
