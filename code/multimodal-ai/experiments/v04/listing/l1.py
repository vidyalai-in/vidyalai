from PIL import Image
from transformers import AutoProcessor

img = Image.new("RGB", (896, 896), "white")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}]}]
for edge in (512, 1024, 2048):
    proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", size={"longest_edge": edge})
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    x = proc(text=[prompt], images=[img], return_tensors="pt")
    n = (x["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<image>")).sum().item()
    print(edge, tuple(x["pixel_values"].shape), n)
