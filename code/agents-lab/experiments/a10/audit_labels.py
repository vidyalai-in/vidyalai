"""Hand labels for the 60 sampled citations (see audit.jsonl for the claims and pages).
page: supported / partly / not supported / unreachable (judged against the live page, fetched 2026-10-25)
span: True if the 150-character cited_text itself contains the claimed fact."""
import json, collections, math
U, S, P = "unreachable", "supported", "partly"
LABELS = {
 0:(U,False,"plus500: HTTP 429 to curl and WebFetch"), 1:(S,True,""), 2:(S,True,""),
 3:(U,True,"fifa.com article renders with JavaScript; plain fetch returns an empty page"),
 4:(S,True,""), 5:(S,True,""), 6:(S,True,"NPR: 76-year-old poet"), 7:(S,False,"press release lists Humboldt; span is the award sentence"),
 8:(S,False,"span is the page title"), 9:(S,False,"press release lists Wuerzburg; span is the award sentence"),
 10:(S,True,""), 11:(S,True,""), 12:(S,True,""), 13:(S,True,""), 14:(U,False,"britannica 403"), 15:(S,True,""),
 16:(U,False,"britannica 403; span describes Halzen, claim is about Kagan"), 17:(S,False,"span truncated before 9/16"),
 18:(U,True,"advisorperspectives 403"), 19:(U,False,"britannica 403; span describes Halzen, claim is about Kagan"),
 20:(S,True,"date matches; the quoted relative age now reads '2 months and 1 week ago'"), 21:(S,True,""),
 22:(S,False,"PEP lists 3.15.0 final: Friday, 2026-10-09; span shows beta dates"),
 23:(S,False,"press release says 'Canadian author'; span lacks it"), 24:(S,False,"span is the page header"),
 25:(S,True,""), 26:(S,True,""), 27:(S,False,""), 28:(S,False,"span is the infobox score, not the date"), 29:(S,True,""),
 30:(S,True,""), 31:(S,True,""), 32:(U,True,"ua.news 403"), 33:(S,False,"releases.rs lists 1.96.0; span is the page top"),
 34:(S,True,""), 35:(S,True,"python.org serves compressed HTML; read after decompressing"), 36:(S,False,"span cut before Stanford"),
 37:(S,True,""), 38:(S,True,"fomcdiff.com says the June statement has no voting paragraph; the Fed's own June 17 statement opens with a 12-0 vote"),
 39:(S,True,""), 40:(S,False,"span is the page header"), 41:(S,False,"span is the page header"),
 42:(S,False,"facts page has Born: 1955, Chicopee; span is the prize motivation"), 43:(S,True,""), 44:(S,True,""), 45:(S,True,""),
 46:(U,True,"fifa.com JavaScript-only page"), 47:(S,False,"span cut before Stanford"), 48:(S,False,"span is the page intro"),
 49:(S,False,"span is the start of Torvalds' mail"), 50:(S,True,""), 51:(S,True,""), 52:(U,True,"britannica 403"), 53:(S,True,""),
 54:(U,True,"whatsnew.fyi 429"), 55:(S,True,""), 56:(S,True,""), 57:(S,False,"page has Apr 16, 2026; span lists download counts"),
 58:(P,False,"an rc6 article that projects rc7 for October 11; it doesn't report that rc7 shipped"), 59:(S,True,""),
}
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (100*(c-h)/d, 100*(c+h)/d)
rows = [json.loads(l) for l in open("audit.jsonl")]
for i, r in enumerate(rows):
    r["label"], r["span_ok"], r["note"] = LABELS[i]
json.dump(rows, open("audit_labeled.json", "w"), indent=1)
by = collections.defaultdict(list)
for r in rows: by[(r["arm"], r["model"])].append(r)
by[("ALL", "")] = rows
for k, v in by.items():
    c = collections.Counter(r["label"] for r in v); reach = [r for r in v if r["label"] != U]
    s = sum(r["label"] == S for r in reach)
    print(k, len(v), dict(c), f"supported among reachable {s}/{len(reach)} {wilson(s,len(reach)) if reach else ''}",
          "span ok", sum(r["span_ok"] for r in v), "/", len(v))
print("primary-domain citations:", sum(any(d in r['url'] for d in ['nobelprize.org','python.org','rust-lang.org','go.dev','openjdk.org','postgresql.org','kernel.org','federalreserve.gov']) for r in rows), "of", len(rows))
