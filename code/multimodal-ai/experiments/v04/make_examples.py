"""Shown images: one read item (224 px vs 896 px) and one dense-count item, rendered by lab/vlm.py."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import vlm
from PIL import Image, ImageDraw
items = {i.id: i for i in vlm.make_items(150)}
OUT = "/home/amit/AlpsCode/ai/vidyalai/public/diagrams/"

it = items["read-022"]
x, y = it.meta["x"], it.meta["y"]
box = (x - 100, y - 60, x + 260, y + 92)                     # label region in master (896) coordinates
big = vlm.render(it, 896).crop(box)
small = vlm.render(it, 224)
sb = tuple(int(round(v / 4)) for v in box)
small_crop = small.crop(sb).resize(big.size, Image.NEAREST)  # the 224 px pixels, enlarged 4x
W = big.width * 2 + 60
canvas = Image.new("RGB", (W, big.height + 70), "white")
d = ImageDraw.Draw(canvas)
f = vlm._font(18)
canvas.paste(small_crop, (20, 50)); canvas.paste(big, (big.width + 40, 50))
d.text((20, 14), "224 px image (enlarged 4x)", font=f, fill=(20, 20, 20))
d.text((big.width + 40, 14), "896 px image", font=f, fill=(20, 20, 20))
d.rectangle([20, 50, 20 + big.width, 50 + big.height], outline=(150, 150, 150))
d.rectangle([big.width + 40, 50, 2 * big.width + 40, 50 + big.height], outline=(150, 150, 150))
canvas.save(OUT + "v04-read-example.png", optimize=True)
print("read", it.answer, canvas.size)

it = items["dense-003"]
img = vlm.render(it, 448)
img.save(OUT + "v04-dense-example.png", optimize=True)
print("dense", it.question, it.answer)
