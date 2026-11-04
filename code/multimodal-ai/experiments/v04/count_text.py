"""Free count_tokens for every distinct text prompt, so billed image tokens = usage.input_tokens - text tokens."""
import json, sys
sys.argv = ["x", "none"]
import claude_run as C
out = {}
for model in ("claude-haiku-5-5", "claude-sonnet-5-5"):
    for terse in ("", C.TERSE or " Do not list, count aloud or explain: reply with the number immediately."):
        for it in C.items:
            q = it.question + " " + C.vlm.SUFFIX[it.task] + terse
            k = f"{model}|{q}"
            if k in out:
                continue
            out[k] = C.cl.messages.count_tokens(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": q}]}]).input_tokens
print(len(out), "prompts")
json.dump(out, open("text_tokens.json", "w"), indent=0)
