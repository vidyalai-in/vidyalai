"""Write measured data into diagrams/v09-wer-by-model.py (DATA line)."""
import json, re
R = json.load(open("/home/amit/.cache/vidyalai-scratch/v09/runs/results.json"))
D = {}
for m in ("tiny", "base", "small", "turbo"):
    e = R["exp1"][f"{m}_b1"]
    D[m] = {"rtf": round(e["meta"]["test-clean"]["audio_s"] / (e["meta"]["test-clean"]["feat_s"] + e["meta"]["test-clean"]["gen_s"]), 1),
            "wer": {sp: [round(e[sp]["english"][k] * 100, 2) for k in ("wer", "lo", "hi")] for sp in ("test-clean", "test-other")}}
p = "/home/amit/AlpsCode/ai/vidyalai/diagrams/v09-wer-by-model.py"
s = open(p).read()
s = re.sub(r"^DATA = .*$", "DATA = " + json.dumps(D), s, flags=re.M)
open(p, "w").write(s)
print(D)
