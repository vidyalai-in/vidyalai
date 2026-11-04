import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, own_eval as oe
cl = ledger.client(); arts = oe.load_articles(); out = {}
for n, a in arts.items():
    p = oe.answer_request({"question": "x"}, a, "off")
    out[n] = cl.messages.count_tokens(model=p["model"], system=p["system"], messages=p["messages"]).input_tokens
print(out); json.dump(out, open("data/article_tokens.json", "w"))
