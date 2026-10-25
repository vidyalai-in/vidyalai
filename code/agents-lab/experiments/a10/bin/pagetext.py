"""Fetch a URL with a browser user agent and print a window of its visible text.
usage: pagetext.py URL [needle] [width]"""
import html, re, sys, urllib.request, hashlib, pathlib
url = sys.argv[1]; needle = sys.argv[2] if len(sys.argv) > 2 else None
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
d = pathlib.Path("/home/amit/.cache/vidyalai-scratch/a10/pages"); d.mkdir(exist_ok=True)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"})
raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
(d / (hashlib.md5(url.encode()).hexdigest()[:10] + ".html")).write_text(raw)
t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
if needle:
    for m in list(re.finditer(re.escape(needle), t, re.I))[:3]:
        print("...", t[max(0, m.start() - width // 3): m.start() + width], "...\n")
else:
    print(t[:width])
