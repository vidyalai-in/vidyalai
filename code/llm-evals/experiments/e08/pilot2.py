import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e08/scripts")
import api
m5 = json.load(open(api.D + "e08-math500.json"))
hard = [i for i, x in enumerate(m5) if x["level"] == 5][:4]
for eff in ("medium", "high"):
    for i in hard:
        r = api.single(api.math_params(m5[i], thinking=api.ADAPTIVE, effort=eff, max_tokens=16000), f"pilot math adaptive {eff}", "pilot.jsonl")
        print(eff, i, r["stop"], r["usage"]["output_tokens"], r["types"], repr(r["text"][-60:]), m5[i]["answer"], flush=True)
