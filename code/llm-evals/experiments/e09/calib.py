import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc
cl = ledger.client()
paras = lc.load_corpus()
tot = sum(len(p) + 2 for p in paras)
print("paragraphs", len(paras), "chars", tot)
out = {}
for n in [20000, 200000]:
    t = lc.filler(paras, n, 3000)
    r = cl.messages.count_tokens(model="claude-haiku-5-5", messages=[{"role": "user", "content": t}])
    out[n] = r.input_tokens
    print(n, r.input_tokens, n / r.input_tokens)
h = lc.build_haystack(paras, 20000, 100, seed=1)
r = cl.messages.count_tokens(model="claude-haiku-5-5", messages=lc.claude_params(h["text"], lc.questions(h)[0][2])["messages"])
print("hay 20000 chars ->", r.input_tokens)
print([ (c["name"], round(c["depth"],3)) for c in h["chars"]])
print([ [round(d,3) for d in c["depths"]] for c in h["chains"]])
i = h["text"].find(h["chars"][2]["needle"]); print(repr(h["text"][i-200:i+150]))
