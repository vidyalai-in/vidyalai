import importlib.util, sys
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from playwright.sync_api import sync_playwright
srv = S.serve(0); port = srv.server_address[1]
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={"width":1280,"height":800})
    pg = ctx.new_page()
    for path in sys.argv[1:]:
        pg.goto(f"http://127.0.0.1:{port}{path}")
        pg.screenshot(path=f"/home/amit/.cache/vidyalai-scratch/a09/shots/{path.strip('/').replace('/','_') or 'home'}.png")
    pg.goto(f"http://127.0.0.1:{port}/travel"); pg.click("[aria-label='Open calendar for departure date']"); pg.screenshot(path="/home/amit/.cache/vidyalai-scratch/a09/shots/cal.png")
    b.close()
