"""Selection on noise: N variants with the same true accuracy p, each scored on its own
random draw of n items; report the expected best score minus p (Monte Carlo, 200,000 trials)."""
import numpy as np
rng = np.random.default_rng(0)
p, n = 0.80, 1000
for N in (1, 2, 5, 10, 27):
    best = rng.binomial(n, p, size=(200_000, N)).max(1) / n
    print(f"N={N:3d}  mean best {100*best.mean():.2f}%  inflation {100*(best.mean()-p):+.2f} pts  "
          f"95% of trials below {100*np.percentile(best,95):.1f}%")
print("SE one run:", round(100*np.sqrt(p*(1-p)/n), 2), "pts")
