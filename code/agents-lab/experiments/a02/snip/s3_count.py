import ledger                      # Part 1's budget wrapper; client() returns an anthropic.Anthropic
from toollib import TOOLS          # 20 tool definitions, 2-4 parameters each

cl = ledger.client()
msg = [{"role": "user", "content": "What's the weather in Paris?"}]
for n in (0, 1, 5, 20):
    extra = {"tools": TOOLS[:n]} if n else {}
    print(n, cl.messages.count_tokens(model="claude-haiku-5-5", messages=msg, **extra).input_tokens)
