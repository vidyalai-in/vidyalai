import importlib.util, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import computer as C
from playwright.sync_api import sync_playwright
srv = S.serve(0); base = f"http://127.0.0.1:{srv.server_address[1]}"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = C.Page(b, base, "x", "/inventory?page=2"); ex = C.BrowserExecutor(pg)
    nodes = ex.cdp.send("Accessibility.getFullAXTree")["nodes"]
    print({n["role"]["value"] for n in nodes if "Warning" in str(n.get("name", {}))}, [n.get("name", {}).get("value") for n in nodes if "Warning" in str(n.get("name", {}))])
    if len(sys.argv) > 1: print("\n".join(l for l in ex.run("read_page", {})[0]["text"].splitlines() if "2970" in l or "Warn" in l))
    b.close()
