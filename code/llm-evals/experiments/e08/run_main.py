"""Full runs: HumanEval x10 (Haiku, thinking off) and MATH-500 x2 for each of two settings."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e08/scripts")
import api
which = sys.argv[1]
he = json.load(open(api.D + "e08-humaneval.json")); m5 = json.load(open(api.D + "e08-math500.json"))
if which == "he":
    reqs = [(f"he{i:03d}-s{s}", api.code_params(t)) for s in range(10) for i, t in enumerate(he)]
    api.run_batch(reqs, api.HAIKU, "humaneval haiku off n=10", "he_haiku")
elif which == "math":
    # one batch per setting and repeat, in sequence, so each worst-case estimate fits the sub-cap
    for r in range(2):
        for name in ("off", "low"):
            if name == "off":
                reqs = [(f"off-r{r}-{i:03d}", api.math_params(x)) for i, x in enumerate(m5)]
            else:
                reqs = [(f"low-r{r}-{i:03d}", api.math_params(x, thinking=api.ADAPTIVE, effort="low",
                                                             max_tokens=8192)) for i, x in enumerate(m5)]
            api.run_batch(reqs, api.HAIKU, f"math500 haiku {name} r{r}", f"math_haiku_{name}_r{r}")
