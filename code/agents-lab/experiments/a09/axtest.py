import importlib.util, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import computer as C
from playwright.sync_api import sync_playwright
srv = S.serve(0); base = f"http://127.0.0.1:{srv.server_address[1]}"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = C.Page(b, base, "t1", sys.argv[1])
    ex = C.BrowserExecutor(pg)
    print(ex.run("read_page", {})[0]["text"])
    for cmd in sys.argv[2:]:
        name, _, js = cmd.partition(":")
        import json
        out = ex.run(name, json.loads(js) if js else {})
        print(">>>", cmd, [o.get("text", o.get("type")) for o in out])
    print(S.SUBMISSIONS)
    b.close()
