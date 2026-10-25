import importlib.util
spec = importlib.util.spec_from_file_location("lsite", "/home/amit/.cache/vidyalai-scratch/a09/lsite.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from playwright.sync_api import sync_playwright
srv = S.serve(0); base = f"http://127.0.0.1:{srv.server_address[1]}"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":1280,"height":800})
    pg.goto(base + "/policies/travel"); print("table top", pg.evaluate("document.querySelectorAll('table')[0].getBoundingClientRect().top"))
    pg.goto(base + "/reports/budget"); print("budget font", pg.evaluate("getComputedStyle(document.querySelector('td')).fontSize"))
    b.close()
