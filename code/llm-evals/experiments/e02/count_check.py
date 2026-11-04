"""Compare Sonnet 5.5 count_tokens with Haiku's billed input tokens on 20 prompts (count_tokens is free)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, items
data = items.load_redux(); cl = ledger.client()
hin = {}
for l in open("runs/haiku-k3.jsonl"):
    r = json.loads(l); hin[int(r["custom_id"][1:5])] = r["usage"]["input_tokens"]
rat = []
for i in range(0, 5700, 285):
    n = cl.messages.count_tokens(model="claude-sonnet-5-5", messages=[{"role": "user", "content": items.mcq_prompt(data[i])}]).input_tokens
    rat.append(n / hin[i]); print(i, n, hin[i])
print("max ratio", max(rat))
