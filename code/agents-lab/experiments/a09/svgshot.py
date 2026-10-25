import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 800, "height": 490})
    pg.goto("file://" + sys.argv[1]); pg.screenshot(path=sys.argv[2]); b.close()
