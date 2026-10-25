# Measured on the RTX 4060 Laptop GPU: parameters, tokens/s, mean GPU power in watts
runs = {
    "22M":  (22_025_088, 103_618, 49.9),
    "47M":  (47_195_712,  49_459, 49.8),
    "88M":  (88_096_512,  29_508, 49.5),
    "126M": (125_854_464, 19_096, 46.0),
}
PRICE = 0.20  # illustrative electricity price, $ per kWh

for name, (N, tok_s, watts) in runs.items():
    D = 20 * N                        # Chinchilla: about 20 tokens per parameter
    C = 6 * N * D                     # training FLOPs, 6ND
    hours = D / tok_s / 3600
    kwh = watts * hours / 1000
    night = tok_s * 8 * 3600 / N      # tokens per parameter reachable in 8 h
    weekend = tok_s * 48 * 3600 / N
    print(f"{name:>4}: D={D/1e9:.2f}B  C={C:.1e}  {hours:4.1f} h  "
          f"{kwh:.2f} kWh ${kwh * PRICE:.2f}  night {night:3.0f}/param  weekend {weekend:4.0f}/param")
