import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
weather = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TOOL_CALL = tok.convert_tokens_to_ids("<tool_call>")  # one token, id 151657

@torch.no_grad()
def p_tool_call(question):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tools=[weather],
                                     add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids
    probs = model(ids).logits[0, -1].float().softmax(-1)  # distribution over the next token
    return probs[TOOL_CALL].item()

for q in ["What's the weather in Paris?", "Should I bring an umbrella in London today?", "Write a haiku about autumn."]:
    print(f"{p_tool_call(q):.2e}  {q}")
