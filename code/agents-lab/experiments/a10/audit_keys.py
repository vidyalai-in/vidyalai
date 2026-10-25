"""For each audited citation, look for the claim's key facts on the live page (normalized text)."""
import hashlib, html, json, re, sys
from pathlib import Path
KEYS = {0:["3.75","raising"],1:["scheduled for friday, october 9"],2:["kashkari, and lorie k. logan"],
 3:["1-0 victory over argentina","second"],4:["oct. 1, 2026"],5:["released august 19"],6:["76-year-old"],
 7:["humboldt"],8:["karl deisseroth, peter hegemann and georg nagel"],9:["würzburg","wurzburg"],
 10:["affiliation at the time of the award: university of wisconsin"],11:["9–3","9-3","three members dissenting"],
 12:["released on october 18"],13:["outshooting argentina 20-3"],14:["d.h. chen"],15:["2026/09/15"],16:["boulogne"],
 17:["9/16","general availability"],18:["3.75%"],19:["boulogne"],20:["19 aug 2026"],21:["october 9, 2026"],
 22:["2026-10-09","3.15.0 final"],23:["canadian","bold and inventive"],24:["python 3.15.0, released on 9 october 2026"],
 25:["9-3"],26:["october 18, 2026"],27:["humboldt university of berlin"],28:["july 19, 2026"],29:["1.93.0"],
 30:["october 21, 2026"],31:["7.3 kernel"],32:["woodford"],33:["1.96.0"],34:["announcing rust 1.95.0","apr. 16"],
 35:["oct. 21, 2026"],36:["howard hughes medical institute","stanford"],37:["decisive contributions to the icecube"],
 38:["no voting paragraph"],39:["19 august 2026"],40:["9 october 2026"],41:["9 october 2026"],42:["chicopee"],
 43:["november 12, 2026"],44:["november 19, 2026"],45:["wcslen"],46:["1-0 victory over argentina after extra-time"],
 47:["howard hughes medical institute"],48:["18 oct 2026"],49:["7.3 kernel","october 18, 2026"],50:["kagan and kenso soai"],
 51:["oct. 15, 2026"],52:["hiroshima"],53:["oct. 9, 2026"],54:["1.97.0"],55:["5 march, 2026"],56:["22 january, 2026"],
 57:["apr 16, 2026"],58:["rc7","october 11"],59:["1 october, 2026"]}
PAGES = Path("pages")
def text_of(url):
    p = PAGES / (hashlib.md5(url.encode()).hexdigest()[:10] + ".html")
    if not p.exists(): return ""
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", p.read_text(), flags=re.S | re.I)
    t = re.sub(r"[#*_`>\[\]]", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    return re.sub(r"\s+", " ", t).lower()
rows = [json.loads(l) for l in open("audit.jsonl")]
for i, a in enumerate(rows):
    t = text_of(a["url"]); hits = []
    for k in KEYS[i]:
        j = t.find(k)
        hits.append(f"{k!r}:{'Y' if j>=0 else 'n'}")
    print(f"#{i} {a['status']} {' '.join(hits)}")
    if len(sys.argv) > 1 and str(i) in sys.argv[1:]:
        for k in KEYS[i]:
            for m in list(re.finditer(re.escape(k), t))[:2]:
                print("    ...", t[max(0, m.start()-250): m.start()+250])
