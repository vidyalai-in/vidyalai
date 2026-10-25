"""Free token counts: toolset overhead, and visual tokens per screenshot at several resolutions."""
import sys, io, base64, json, math
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger, computer as C
from PIL import Image
cl = ledger.client()
out = {}
for model in ["claude-haiku-5-5", "claude-sonnet-5-5"]:
    msg = [{"role": "user", "content": "Open the travel page."}]
    base = cl.messages.count_tokens(model=model, messages=msg).input_tokens
    r = {"no tools": base}
    for name, entry in [("computer (default)", C.computer_toolset()), ("computer, no zoom", C.computer_toolset(zoom=False)),
                        ("browser (default 27)", {"type": "browser_toolset_20260801"}),
                        ("browser, no tabs (hybrid arm)", C.browser_toolset(True)),
                        ("browser, no tabs/screenshot/zoom (text arm)", C.browser_toolset(False))]:
        r[name] = cl.messages.count_tokens(model=model, messages=msg, tools=[entry]).input_tokens - base
    out[model] = r
    print(model, r)
# image tokens: real screenshot of the intranet resized to each resolution
src = Image.open("/home/amit/.cache/vidyalai-scratch/a09/shots/inventory.png").convert("RGB")
rows = []
for w, h in [(1024, 768), (1280, 720), (1280, 800), (1366, 768), (1920, 1080), (2560, 1440)]:
    buf = io.BytesIO(); src.resize((w, h)).save(buf, "PNG")
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}
    for model in ["claude-haiku-5-5"]:
        t0 = cl.messages.count_tokens(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": "Describe."}]}]).input_tokens
        t1 = cl.messages.count_tokens(model=model, messages=[{"role": "user", "content": [img, {"type": "text", "text": "Describe."}]}]).input_tokens
        formula = math.ceil(w / 28) * math.ceil(h / 28)
        rows.append(dict(w=w, h=h, model=model, measured=t1 - t0, formula=formula, png_kb=len(buf.getvalue()) // 1024))
        print(rows[-1])
json.dump({"toolsets": out, "images": rows}, open("/home/amit/.cache/vidyalai-scratch/a09/tokcount.json", "w"), indent=1)
