"""Score Part 16 runs against the seeded ground truth.

python score.py <config> [<config> ...]

A finding matches a seed when it is on the same page within 3 lines of the seeded line (one finding
per seed per run; extra matches are duplicates). Its fix is correct when, applied to the seeded page:
  link  the seeded broken reference is gone and no new broken reference appears
  code  the failing example near the seed passes and no new example fails
  fact  the edited lines no longer say the wrong thing (`bad`) and say the right thing (`good`)
Unmatched findings are looked up in labels.json (hand labels: real pre-existing defect or false positive).
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import evals  # noqa: E402
import finale  # noqa: E402
from seeds import SEEDS, HARD, CORPUS  # noqa: E402

A16 = Path(__file__).parent
WINDOW = 3
LABELS = json.loads((A16 / "labels.json").read_text()) if (A16 / "labels.json").exists() else []
_cache = {}


def checks(corpus, page, text=None):
    key = (str(corpus), page, text)
    if key not in _cache:
        if text is None:
            _cache[key] = (finale.link_report(corpus, page), finale.example_report(corpus, page))
        else:
            tmp = A16 / "tmp_fix.rst"
            tmp.write_text(text)
            _cache[key] = (finale.link_report(corpus, page, (tmp, page)),
                           finale.example_report(corpus, page, (tmp, page)))
    return _cache[key]


def fix_ok(seed, f, corpus):
    page = seed["file"]
    text = (corpus / "docs" / page).read_text()
    if text.count(f["old_text"]) != 1:
        return False
    new = text.replace(f["old_text"], f["new_text"])
    bl, be = checks(corpus, page)
    al, ae = checks(corpus, page, new)
    near = lambda x: abs(x["line"] - seed["line"]) <= WINDOW  # noqa: E731
    if seed["kind"] == "link":
        return not any(near(b) for b in al["broken"]) and len(al["broken"]) < len(bl["broken"])
    if seed["kind"] == "code":
        return not any(near(b) for b in ae["failures"]) and len(ae["failures"]) < len(be["failures"])
    # fact: look at the edited region of the new text
    lines = new.splitlines()
    region = "\n".join(lines[max(0, seed["line"] - 3): seed["line"] + 2])
    no_new_breaks = len(al["broken"]) <= len(bl["broken"]) and len(ae["failures"]) <= len(be["failures"])
    return (not re.search(seed["bad"], region)) and bool(re.search(seed["good"], region)) and no_new_breaks


def corpus_of(cfg):
    return "clean" if "clean" in cfg else "hard" if "hard" in cfg else "seeded"


def seeds_of(cfg):
    return {"clean": [], "hard": HARD, "seeded": SEEDS}[corpus_of(cfg)]


def label_of(page, f):
    for lab in LABELS:
        if (lab["page"] == page and lab["lo"] <= f["line"] <= lab["hi"] and lab.get("kind") in (None, f["kind"])
                and lab.get("contains", "") in f["problem"] + f["old_text"]):
            return lab
    return None


def score(cfg):
    runs = [json.loads(p.read_text()) for p in sorted((A16 / "runs" / cfg).glob("*.json"))]
    corpus = A16 / "corpus" / corpus_of(cfg)
    seeds_by_page = defaultdict(list)
    for s in seeds_of(cfg):
        seeds_by_page[s["file"]].append(s)
    found = defaultdict(list)       # seed id -> [bool per run]
    fixed = defaultdict(list)       # seed id -> [bool per found run]
    unmatched, dup = [], 0
    real_hits, fp = Counter(), Counter()
    for r in runs:
        page = r["page"]
        hit = {}
        for f in r["findings"]:
            cands = [s for s in seeds_by_page[page] if abs(f["line"] - s["line"]) <= WINDOW and s["id"] not in hit]
            cands.sort(key=lambda s: (s["kind"] != f["kind"], abs(f["line"] - s["line"])))
            if cands:
                hit[cands[0]["id"]] = f
                continue
            if any(abs(f["line"] - s["line"]) <= WINDOW for s in seeds_by_page[page]):
                dup += 1
                continue
            lab = label_of(page, f)
            unmatched.append((page, f, lab))
            if lab and lab["label"] == "real":
                real_hits[lab["id"]] += 1
            else:
                fp[(page, f["line"], f["kind"])] += 1
        for s in seeds_by_page[page]:
            found[s["id"]].append(s["id"] in hit)
            if s["id"] in hit:
                fixed[s["id"]].append(fix_ok(s, hit[s["id"]], corpus))
    return runs, found, fixed, unmatched, dup, real_hits, fp


def ci(k, n):
    lo, hi = evals.wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]" if n else "n/a"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def report(cfg):
    runs, found, fixed, unmatched, dup, real_hits, fp = score(cfg)
    print(f"\n===== {cfg}: {len(runs)} runs")
    stops = Counter(r["stop"] for r in runs)
    print("stops", dict(stops))
    for kind in ("link", "code", "fact"):
        ids = [s["id"] for s in seeds_of(cfg) if s["kind"] == kind]
        k = sum(sum(found[i]) for i in ids)
        n = sum(len(found[i]) for i in ids)
        fk = sum(sum(fixed[i]) for i in ids)
        fn = sum(len(fixed[i]) for i in ids)
        boot = evals.task_bootstrap({i: found[i] for i in ids}, lambda R: sum(sum(v) for v in R.values()) / max(1, sum(len(v) for v in R.values()))) if n else (0, 0)
        print(f"  {kind}: found {ci(k, n)}  (seed-bootstrap [{100*boot[0]:.1f}, {100*boot[1]:.1f}])   fix correct {ci(fk, fn)}")
    allk = sum(sum(v) for v in found.values()); alln = sum(len(v) for v in found.values())
    allf = sum(sum(v) for v in fixed.values()); allfn = sum(len(v) for v in fixed.values())
    if alln:
        print(f"  all seeds: found {ci(allk, alln)}; fix correct {ci(allf, allfn)}")
        always = sum(all(v) for v in found.values()); never = sum(not any(v) for v in found.values())
        print(f"  seeds found in every run: {always}/{len(found)}; never found: {never}/{len(found)} -> {[i for i, v in found.items() if not any(v)]}")
        print("  per seed:", {i: f"{sum(v)}/{len(v)}" for i, v in found.items()})
        print("  per seed fixed:", {i: f"{sum(v)}/{len(v)}" for i, v in fixed.items()})
    nf = sum(len(r["findings"]) for r in runs)
    nreal = sum(real_hits.values()); nfp = sum(fp.values())
    print(f"  findings {nf}: seeded TP {allk}, duplicates {dup}, real pre-existing {nreal}, false positive/unlabeled {nfp}")
    if nf:
        print(f"  precision (seeded + real) {ci(allk + nreal, allk + nreal + nfp)}")
    print("  real pre-existing hits:", dict(real_hits))
    cost = sum(r["cost"] for r in runs)
    walls = [r["wall_s"] for r in runs]
    print(f"  cost ${cost:.4f}; per page-run ${cost / len(runs):.5f}; per true defect found ${cost / max(1, allk + nreal):.5f}")
    print(f"  wall per page: median {statistics.median(walls):.1f}s p95 {pct(walls, 0.95):.1f}s; "
          f"steps median {statistics.median([r['steps'] for r in runs])}, max {max(r['steps'] for r in runs)}; "
          f"tool calls {sum(r['tool_calls'] for r in runs)}; last-step context median {statistics.median([r['last_context'] for r in runs])}, "
          f"max {max(r['last_context'] for r in runs)}")
    dec = Counter(d for r in runs for _, d in r["decisions"])
    tools = Counter(n for r in runs for n, _ in r["decisions"])
    print("  policy decisions", dict(dec), "tool use", dict(tools))
    print("  replaced (idempotent re-reports):", sum(f.get("replaced", 0) for r in runs for f in r["findings"]))
    return unmatched


if __name__ == "__main__":
    for cfg in [a for a in sys.argv[1:] if not a.startswith("-")]:
        um = report(cfg)
        if "-v" in sys.argv:
            for page, f, lab in um:
                print("   UNMATCHED", page, f["line"], f["kind"], (lab or {}).get("id", "??"), "|", f["problem"][:200])
