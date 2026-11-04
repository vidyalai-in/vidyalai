import json, sys
sys.argv = ["x", "none"]
import claude_run as C
it = [i for i in C.items if i.id == "count-000"][0]
q = it.question + " " + C.vlm.SUFFIX[it.task]
m = C.ledger.create(C.cl, "v04", note="v04 text-only check sonnet", model="claude-sonnet-5-5", max_tokens=16,
                    thinking={"type": "between_tools"}, messages=[{"role": "user", "content": [{"type": "text", "text": q}]}])
n = C.cl.messages.count_tokens(model="claude-sonnet-5-5", messages=[{"role": "user", "content": [{"type": "text", "text": q}]}]).input_tokens
print(json.dumps(dict(billed=m.usage.input_tokens, counted=n, text="".join(c.text for c in m.content if c.type == "text"))))
open("textonly_check_sonnet.json", "w").write(json.dumps(dict(billed=m.usage.input_tokens, counted=n)))
