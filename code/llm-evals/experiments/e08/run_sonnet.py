"""Sonnet 5.5 on HumanEval, 2 samples per problem, lowest thinking setting ("between_tools"), default effort."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/e08/scripts")
import api
he = json.load(open(api.D + "e08-humaneval.json"))
if sys.argv[1] == "pilot":
    for i in (0, 32):
        r = api.single(api.code_params(he[i], model=api.SONNET, thinking=api.SONNET_LOW), "pilot humaneval sonnet", "pilot.jsonl")
        print(r["stop"], r["usage"]["input_tokens"], r["usage"]["output_tokens"], r["types"], r["text"][-150:])
else:
    s = int(sys.argv[1])
    reqs = [(f"he{i:03d}-s{s}", api.code_params(t, model=api.SONNET, thinking=api.SONNET_LOW)) for i, t in enumerate(he)]
    api.run_batch(reqs, api.SONNET, f"humaneval sonnet s{s}", f"he_sonnet_s{s}")
