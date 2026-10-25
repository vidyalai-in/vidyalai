"""Part 16 baseline with no model: the two checkers alone, plus a rule-based fixer.

Link fix rule: replace the broken target with the checker's closest match (if there is one).
Code fix rule: replace the example's expected output with what the code printed (the "accept the
new output" rule many doctest tools offer).
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import finale  # noqa: E402
from score import fix_ok, WINDOW  # noqa: E402
from seeds import SEEDS, CORPUS  # noqa: E402

A16 = Path(__file__).parent
corpus = A16 / "corpus" / "seeded"
flags, found, fixed, extra = 0, {}, {}, []
for page in CORPUS:
    links, ex = finale.link_report(corpus, page), finale.example_report(corpus, page)
    text = (corpus / "docs" / page).read_text()
    items = [("link", b["line"], b) for b in links["broken"]] + [("code", f["line"], f) for f in ex["failures"]]
    seen = set()
    for kind, line, b in items:
        flags += 1
        seed = next((s for s in SEEDS if s["file"] == page and abs(s["line"] - line) <= WINDOW and s["id"] not in seen), None)
        if kind == "link":
            old = b["target"].lstrip("~")
            new = b["close_matches"][0] if b["close_matches"] else None
            fix = dict(old_text=old, new_text=new) if new else None
        else:
            # replace the expected-output lines under the example with the actual output
            lines = text.split("\n")
            i = b["line"]  # 1-based line of the >>> source; expected output starts on the next line
            while lines[i].lstrip().startswith("..."):  # continuation lines of the source
                i += 1
            n = len(b["expected"].splitlines())
            old_block = "\n".join(lines[i - 1:i + n])
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            new_block = "\n".join([lines[i - 1]] + [indent + g for g in b["got"].splitlines()])
            fix = dict(old_text=old_block, new_text=new_block) if n and text.count(old_block) == 1 else None
        if seed is None and any(s["file"] == page and abs(s["line"] - line) <= WINDOW for s in SEEDS):
            continue  # a follow-on failure of a seed already counted
        if seed is None:
            extra.append((page, line, kind, b))
            continue
        seen.add(seed["id"])
        found[seed["id"]] = True
        fixed[seed["id"]] = bool(fix) and fix_ok(seed, fix, corpus)
for kind in ("link", "code", "fact"):
    ids = [s["id"] for s in SEEDS if s["kind"] == kind]
    print(f"{kind}: flagged {sum(found.get(i, False) for i in ids)}/8, rule-fix correct "
          f"{sum(fixed.get(i, False) for i in ids)}/8  {[i for i in ids if found.get(i) and not fixed.get(i)]}")
print("flags", flags, "not matching a seed:", [(p, l, k) for p, l, k, _ in extra])
