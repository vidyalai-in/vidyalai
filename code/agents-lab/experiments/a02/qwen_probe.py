"""Inside an open model: how Qwen3-0.6B renders tools and how likely it is to start a tool call.

Runs on CPU in bfloat16. No API calls.
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.set_num_threads(8)
MID = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16)
model.eval()

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
STOCK = {"type": "function", "function": {"name": "get_stock_price", "description": "Get the latest price for a stock ticker.",
         "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}
TOOLS = [WEATHER, STOCK]
TC = tok.convert_tokens_to_ids("<tool_call>")


def prompt(user, tools=TOOLS):
    return tok.apply_chat_template([{"role": "user", "content": user}], tools=tools, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


@torch.no_grad()
def next_probs(text):
    ids = tok(text, return_tensors="pt").input_ids
    logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits, -1), ids.shape[1]


out = {}
# 1. The rendered prompt
p = prompt("What's the weather in Paris?")
out["rendered"] = p
no_tools = tok.apply_chat_template([{"role": "user", "content": "What's the weather in Paris?"}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
out["tokens_with_2_tools"] = len(tok(p).input_ids)
out["tokens_without_tools"] = len(tok(no_tools).input_ids)
out["tool_call_token_id"] = TC
print(p)
print(out["tokens_with_2_tools"], out["tokens_without_tools"], TC)

# 2. P(first token = <tool_call>) for prompts that need a tool and prompts that don't
NEED = ["What's the weather in Paris?", "Is it raining in Tokyo right now?", "How hot is it in Cairo today?",
        "What's AAPL trading at?", "Give me the current price of NVDA.", "Should I bring an umbrella in London today?",
        "What is Tesla's stock price right now?", "Weather in Mumbai please.", "How cold is it in Oslo?", "Price of MSFT?"]
NONE = ["Write a haiku about autumn.", "What is 17 times 23?", "Translate 'good morning' into Spanish.",
        "Who wrote Pride and Prejudice?", "Explain what a stock ticker is.", "Tell me a joke about cats.",
        "What is the capital of Australia?", "Summarize the plot of Hamlet in one sentence.",
        "What does 'weather' mean as a verb?", "Give me three names for a goldfish."]
rows = []
for kind, qs in (("needs_tool", NEED), ("no_tool", NONE)):
    for q in qs:
        probs, n = next_probs(prompt(q))
        top = torch.topk(probs, 3)
        rows.append({"kind": kind, "q": q, "p_tool_call": probs[TC].item(),
                     "top3": [(tok.decode([i]), round(v.item(), 4)) for v, i in zip(top.values, top.indices)]})
        print(kind, f"{probs[TC].item():.4f}", q, rows[-1]["top3"])
out["probe"] = rows

# 3. Which tool? Probability of each name's first token after '<tool_call>\n{"name": "'
sel = []
for q in ["What's the weather in Paris?", "What's AAPL trading at?", "Is it a good day for a picnic in Rome?"]:
    probs, _ = next_probs(prompt(q) + '<tool_call>\n{"name": "')
    w = tok('get_weather', add_special_tokens=False).input_ids
    s = tok('get_stock_price', add_special_tokens=False).input_ids
    # shared prefix 'get' then the branch; walk to the first differing token
    k = next(i for i in range(min(len(w), len(s))) if w[i] != s[i])
    ctx = prompt(q) + '<tool_call>\n{"name": "' + tok.decode(w[:k])
    pr, _ = next_probs(ctx)
    pw, ps = pr[w[k]].item(), pr[s[k]].item()
    sel.append({"q": q, "branch_tokens": [tok.decode([w[k]]), tok.decode([s[k]])], "p_weather": pw, "p_stock": ps})
    print(q, sel[-1])
out["selection"] = sel

# 4. One greedy generation, verbatim
ids = tok(prompt("What's the weather in Paris?"), return_tensors="pt").input_ids
g = model.generate(ids, max_new_tokens=40, do_sample=False)
out["greedy"] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
out["greedy_ids"] = g[0, ids.shape[1]:].tolist()
print(repr(out["greedy"]))
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/a02/runs/qwen_probe.json", "w"), indent=1)
