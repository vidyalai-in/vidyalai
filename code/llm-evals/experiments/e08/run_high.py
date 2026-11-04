"""MATH-500 once more with adaptive thinking at effort high (where Haiku actually thinks)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e08/scripts")
import api
m5 = json.load(open(api.D + "e08-math500.json"))
reqs = [(f"high-r0-{i:03d}", api.math_params(x, thinking=api.ADAPTIVE, effort="high", max_tokens=8192)) for i, x in enumerate(m5)]
api.run_batch(reqs, api.HAIKU, "math500 haiku adaptive high r0", "math_haiku_high_r0")
