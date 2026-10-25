import importlib.util, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import computer as C
from playwright.sync_api import sync_playwright
srv = S.serve(0); base = f"http://127.0.0.1:{srv.server_address[1]}"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = C.Page(b, base, "t1", "/travel")
    ex = C.ComputerExecutor(pg)
    ex.run("left_click", {"coordinate": [372, 213]})
    open("/home/amit/.cache/vidyalai-scratch/a09/shots/select-open.png","wb").write(pg.png())
    ex.run("left_click", {"coordinate": [360, 300]})
    print(pg.page.eval_on_selector("#destination", "e=>e.value"))
    bx = C.BrowserExecutor(pg); print(bx.run("read_page", {"filter":"interactive"})[0]["text"][-700:])
    b.close()
