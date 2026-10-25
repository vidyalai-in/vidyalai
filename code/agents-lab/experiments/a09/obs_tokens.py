"""Tokens per observation on the same pages: screenshot vs read_page vs get_page_text (free token counting)."""
import importlib.util, sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import computer as C, ledger
from playwright.sync_api import sync_playwright
cl = ledger.client()
def toks(content):
    m = "claude-haiku-5-5"
    base = cl.messages.count_tokens(model=m, messages=[{"role": "user", "content": "x"}]).input_tokens
    return cl.messages.count_tokens(model=m, messages=[{"role": "user", "content": content + [{"type": "text", "text": "x"}]}]).input_tokens - base
srv = S.serve(0); base = f"http://127.0.0.1:{srv.server_address[1]}"
rows = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for path in ["/", "/inventory", "/travel", "/reports/budget", "/policies/travel", "/vendors"]:
        pg = C.Page(b, base, "obs", path); ex = C.BrowserExecutor(pg)
        r = dict(page=path, screenshot=toks([C.image_block(pg.png())]))
        for name, inp in [("read_page", {}), ("read_page_interactive", {"filter": "interactive"}), ("read_page_all", {"filter": "all"}), ("get_page_text", {})]:
            out = ex.run(name.replace("_interactive", "").replace("_all", ""), inp)
            r[name] = toks([{"type": "text", "text": out[0]["text"]}])
        h = pg.page.evaluate("document.documentElement.scrollHeight"); r["height"] = h
        print(r); rows.append(r); pg.close()
    b.close()
json.dump(rows, open("/home/amit/.cache/vidyalai-scratch/a09/obs_tokens.json", "w"), indent=1)
