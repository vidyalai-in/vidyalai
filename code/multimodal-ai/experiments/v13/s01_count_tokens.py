"""Exp 1a (free): image tokens on claude-haiku-5-5 via messages.count_tokens, extending Parts 1 and 4.
(a) 1, 2, 4 and 8 different report pages in one request: is the +3 overhead per image or per request?
(b) the same page as PNG, JPEG and WebP: do file bytes change the token count?
(c) the same page as a PDF with a real text layer (matplotlib) against the PNG.
(d) the prompts used by the caching experiment (S, L, XL), for sizing."""
import base64, io, json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import ledger, serving as S
from vlm import claude_tokens
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/amit/.cache/vidyalai-scratch/v13"
MODEL = "claude-haiku-5-5"
cl = ledger.client()
Q = "On page 1, what is the 'Pallets out' value for South depot? Answer with the number only."


def count(content):
    return cl.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": content}]).input_tokens


def b64(b):
    return base64.standard_b64encode(b).decode()


out = {}
base = count([{"type": "text", "text": Q}])
out["text_only"] = base
# (a) several images
rows = []
for n in (1, 2, 4, 8):
    blocks = [S.image_block(S.report_page(i)) for i in range(n)]
    t = count(blocks + [{"type": "text", "text": Q}])
    rows.append({"images": n, "total": t, "image_tokens": t - base, "formula": n * claude_tokens(1200, 800),
                 "overhead_per_image": (t - base - n * claude_tokens(1200, 800)) / n})
out["multi"] = rows
# (b) formats
img = S.report_page(0)
fm = []
for fmt, mt, kw in (("PNG", "image/png", {"optimize": True}), ("JPEG", "image/jpeg", {"quality": 85}),
                    ("JPEG", "image/jpeg", {"quality": 30}), ("WEBP", "image/webp", {"quality": 80})):
    b = S.png_bytes(img, fmt, **kw)
    t = count([{"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64(b)}},
               {"type": "text", "text": Q}])
    fm.append({"format": fmt, "kw": kw, "bytes": len(b), "image_tokens": t - base})
out["formats"] = fm
# (c) the same page as a PDF with a text layer
f = S.page_facts(0)
fig = plt.figure(figsize=(12, 8), dpi=100)
fig.text(0.033, 0.92, f"Depot operations report, page {f['page']}", fontsize=24, weight="bold")
fig.text(0.033, 0.88, "Synthetic data drawn for a serving experiment", fontsize=13)
ax = fig.add_axes([0.075, 0.2, 0.39, 0.61])
ax.bar(S.MONTHS, f["bars"], color="#3269be"); ax.set_ylim(0, 100); ax.set_title("Trucks unloaded per day", loc="left")
for i, v in enumerate(f["bars"]):
    ax.text(i, v + 1, str(v), ha="center", fontsize=11, weight="bold")
cells = [[r] + [str(v) for v in f["table"][r]] for r in S.ROWS]
ta = fig.add_axes([0.52, 0.42, 0.45, 0.4]); ta.axis("off"); ta.set_title("Weekly depot totals", loc="left")
ta.table(cellText=cells, colLabels=("Depot",) + S.COLS, loc="upper left", cellLoc="left").scale(1, 2)
fig.text(0.033, 0.04, "Page 1 of 8", fontsize=11)
buf = io.BytesIO(); fig.savefig(buf, format="pdf"); pdf = buf.getvalue()
fig.savefig(f"{D}/data/page0.pdf")
t = count([{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64(pdf)}},
           {"type": "text", "text": Q}])
out["pdf"] = {"bytes": len(pdf), "doc_tokens": t - base, "png_tokens": fm[0]["image_tokens"]}
# (d) experiment prompts
SYS = "Answer questions about the attached report pages. Reply with the number only."
def cnt_sys(content):
    return cl.messages.count_tokens(model=MODEL, system=SYS, messages=[{"role": "user", "content": content}]).input_tokens
out["exp_prompts"] = {
    "S": cnt_sys([S.image_block(S.report_page(0)), {"type": "text", "text": Q}]),
    "L": cnt_sys([S.image_block(S.report_page(0, 2352, 1568)), {"type": "text", "text": Q}]),
    "XL": cnt_sys([S.image_block(S.report_page(i, 2352, 1568)) for i in range(8)] + [{"type": "text", "text": Q}]),
}
json.dump(out, open(f"{D}/runs/count_tokens.json", "w"), indent=1)
print(json.dumps(out, indent=1))
