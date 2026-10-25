from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
weather = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

prompt = tok.apply_chat_template(messages, tools=[weather], add_generation_prompt=True,
                                 tokenize=False, enable_thinking=False)
print(prompt)
