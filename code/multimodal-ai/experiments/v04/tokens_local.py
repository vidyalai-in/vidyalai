"""Image tokens per image for the two local VLMs, from their processors alone (CPU, no weights)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vlm
from transformers import AutoProcessor
it = vlm.make_items(5)[0]
rows = []
def count(proc, img, name):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    x = proc(text=[proc.apply_chat_template(msgs, add_generation_prompt=True)], images=[img], return_tensors="pt")
    tid = proc.tokenizer.convert_tokens_to_ids(name)
    return int((x["input_ids"] == tid).sum()), list(x["pixel_values"].shape)
q = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=56 * 56, max_pixels=1344 * 1344)
qd = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")   # library defaults
for size in (224, 448, 896, 1344, 1792):
    img = vlm.render(it, size)
    for tag, p in (("ours", q), ("default", qd)):
        n, shp = count(p, img, "<|image_pad|>")
        rows.append(dict(model="Qwen2-VL-2B", setting=tag, size=size, image_tokens=n, pixel_values=shp))
        print(rows[-1], flush=True)
for le in (512, 1024, 1536, 2048):
    p = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", size={"longest_edge": le})
    for size in (224, 448, 896, 1792):
        n, shp = count(p, vlm.render(it, size), "<image>")
        rows.append(dict(model="SmolVLM-256M", setting=f"longest_edge={le}", size=size, image_tokens=n, pixel_values=shp))
        print(rows[-1], flush=True)
json.dump(rows, open("tokens_local.json", "w"), indent=1)
