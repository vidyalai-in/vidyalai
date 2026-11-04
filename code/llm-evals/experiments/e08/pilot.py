import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e08/scripts")
import api
he = json.load(open(api.D + "e08-humaneval.json")); m5 = json.load(open(api.D + "e08-math500.json"))
for i in (0, 32, 100):
    r = api.single(api.code_params(he[i]), "pilot humaneval", "pilot.jsonl")
    print(he[i]["task_id"], r["stop"], r["usage"]["input_tokens"], r["usage"]["output_tokens"], r["types"]); print(r["text"][:400])
for i in (0, 250):
    r = api.single(api.math_params(m5[i]), "pilot math off", "pilot.jsonl")
    print("math off", r["stop"], r["usage"]["output_tokens"], r["types"], repr(r["text"][-200:]), m5[i]["answer"])
    r = api.single(api.math_params(m5[i], thinking=api.ADAPTIVE, effort="low", max_tokens=16000), "pilot math adaptive low", "pilot.jsonl")
    print("math low", r["stop"], r["usage"]["output_tokens"], r["types"], repr(r["text"][-200:]), m5[i]["answer"])
