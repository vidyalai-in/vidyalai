#!/bin/bash
# Serve one GGUF with llama-server (CPU), ask the reference prompts greedily, then stop the server.
T=/home/amit/.cache/vidyalai-scratch/t15
M=${1:-chat88-q4_k_m}; PORT=8089
$T/llama.cpp/build/bin/llama-server -m $T/gguf/$M.gguf --jinja -c 1024 -t 8 --port $PORT \
  > $T/post/server-$M.log 2>&1 &
PID=$!
for i in $(seq 60); do curl -s localhost:$PORT/health | grep -q ok && break; sleep 0.5; done
curl -s localhost:$PORT/apply-template -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'; echo
python3 - "$PORT" "$M" <<'PY'
import json, sys, urllib.request
port, name = sys.argv[1], sys.argv[2]
prompts = ["What is the capital of France?", "Explain photosynthesis to a ten-year-old in two sentences.",
           "Give me three tips for sleeping better.", "What is 17 + 25?",
           "Write a haiku about the monsoon.", "Why is the sky blue?"]
def ask(msgs, **kw):
    body = {"messages": msgs, "max_tokens": 256, **kw}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
out = []
for p in prompts:
    r = ask([{"role": "user", "content": p}], temperature=0)
    c = r["choices"][0]
    out.append({"prompt": p, "mode": "greedy", "text": c["message"]["content"], "finish": c["finish_reason"],
                "tokens": r["usage"]["completion_tokens"], "timings": r.get("timings")})
    print(json.dumps(out[-1]["text"])[:200], c["finish_reason"], r["usage"]["completion_tokens"])
conv = [{"role": "user", "content": "What is the capital of France?"}]
r = ask(conv, temperature=0); conv += [{"role": "assistant", "content": r["choices"][0]["message"]["content"]},
                                       {"role": "user", "content": "What is it famous for?"}]
r2 = ask(conv, temperature=0)
out.append({"prompt": conv, "mode": "greedy two-turn", "text": r2["choices"][0]["message"]["content"],
            "finish": r2["choices"][0]["finish_reason"], "tokens": r2["usage"]["completion_tokens"]})
print("two-turn:", json.dumps(out[-1]["text"])[:300])
for seed in (0, 1):
    r = ask([{"role": "user", "content": "Why is the sky blue?"}], temperature=0.8, top_k=50, seed=seed)
    out.append({"prompt": "Why is the sky blue?", "mode": f"sampled T=0.8 top_k=50 seed={seed}",
                "text": r["choices"][0]["message"]["content"], "finish": r["choices"][0]["finish_reason"]})
    print("sampled:", json.dumps(out[-1]["text"])[:300])
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/t15/post/chat_{name}.json", "w"), indent=1)
PY
kill $PID; wait $PID 2>/dev/null
curl -s localhost:$PORT/health || echo "server stopped"
