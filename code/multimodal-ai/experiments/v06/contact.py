# Contact sheets for hand labels: seed 11, guidance 1, 7.5 and 20, all 24 prompts, shuffled so the scale is not obvious.
import json, random, textwrap
from PIL import Image, ImageDraw, ImageFont
recs = [json.loads(l) for l in open("runs/guidance.jsonl")]
pick = [r for r in recs if r["seed"] == 11 and r["g"] in (1.0, 7.5, 20.0)]
random.Random(0).shuffle(pick)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
S, C = 300, 4
order = []
for sh in range(0, len(pick), 12):
    grp = pick[sh:sh + 12]
    fig = Image.new("RGB", (C * (S + 8), 3 * (S + 48)), "white"); d = ImageDraw.Draw(fig)
    for i, r in enumerate(grp):
        x, y = (i % C) * (S + 8), (i // C) * (S + 48)
        fig.paste(Image.open(f"runs/img/guidance/{r['key']}.png").resize((S, S)), (x, y + 44))
        d.text((x + 2, y + 2), "\n".join(textwrap.wrap(f"{sh+i}: {r['prompt']}", 36)), fill="black", font=font)
        order.append(r["key"])
    fig.save(f"runs/contact_{sh//12}.jpg", quality=88)
json.dump(order, open("runs/contact_order.json", "w"))
print(len(order))
