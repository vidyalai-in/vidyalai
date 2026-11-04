"""AnimateDiff frame-strip figure. Frames flagged by SD 1.5's safety checker are never shown: they become grey tiles."""
import json, os
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
flags = json.load(open("runs/frames_check_all.json"))
flags["nomotion-p1"]["flag_idx"] = json.load(open("runs/frames_check_nomotion-p1.json"))["nomotion-p1"]["flag_idx"]
rows = [("ad-p1", "AnimateDiff: \"a red fox walking through fresh snow in a pine forest\""),
        ("nomotion-p1", "Same prompt, seed and pipeline with the motion modules switched off"),
        ("ad-p2", "AnimateDiff: \"a campfire burning at night, sparks rising\"")]
size, gap, lab, every = 160, 4, 30, 2
n = len(range(0, 16, every)); W = n * (size + gap) - gap; H = len(rows) * (lab + size + gap)
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); y = 0
for name, title in rows:
    bad = set(flags[name].get("flag_idx", [])) if flags[name]["flagged"] else set()
    assert flags[name]["flagged"] == 0 or bad, name
    d.text((0, y + 6), title, fill=(11, 26, 51), font=FB); y += lab
    for j, k in enumerate(range(0, 16, every)):
        x = j * (size + gap)
        if k in bad:
            d.rectangle([x, y, x + size - 1, y + size - 1], fill=(150, 150, 150))
            d.text((x + size / 2, y + size / 2 - 9), "flagged by the", fill="white", font=FS, anchor="mm")
            d.text((x + size / 2, y + size / 2 + 9), "safety checker", fill="white", font=FS, anchor="mm")
        else:
            im.paste(Image.open(f"frames/{name}/{k:02d}.png").convert("RGB").resize((size, size), Image.LANCZOS), (x, y))
        d.text((x + 5, y + size - 21), f"{k + 1}", fill="white", font=F, stroke_width=2, stroke_fill="black")
    y += size + gap
out = "/home/amit/AlpsCode/ai/vidyalai/public/diagrams/v12-animatediff-frames.webp"
im.save(out, quality=82, method=6); print(im.size, os.path.getsize(out) // 1024, "KB")
