from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width":1280,"height":800})
    pg.set_content("<h1>hi</h1><button>OK</button>")
    print(pg.title(), b.version)
    pg.screenshot(path="t.png")
    print(pg.accessibility.snapshot() if hasattr(pg,'accessibility') else 'no a11y')
    print(pg.locator("body").aria_snapshot())
    b.close()
