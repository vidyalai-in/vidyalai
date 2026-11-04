import sys, json, re, pathlib
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); import ledger
cl = ledger.client()
B = pathlib.Path("/home/amit/AlpsCode/ai/vidyalai/src/content/blog")
out = {}
for f in sorted(B.glob("e0[1-8]-*.md")):
    t = f.read_text()
    fm, body = t.split("\n---\n", 1)[0], t.split("\n---\n", 1)[1]
    title = re.search(r'^title: "(.*)"', fm, re.M).group(1)
    n = cl.messages.count_tokens(model="claude-haiku-5-5", messages=[{"role": "user", "content": body}]).input_tokens
    out[f.stem] = dict(title=title, words=len(body.split()), tokens=n)
    print(f.stem, n, len(body.split()))
print(sum(v["tokens"] for v in out.values()))
