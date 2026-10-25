"""Classify critique replies: did the reviewer claim a real problem, and what happened to the code."""
import json, re, sys
from collections import Counter
NOPROB = re.compile(r"(no (real |clear |correctness |actual |genuine )?(problems|violations|issues|bugs|defects)|"
                    r"function is unchanged|code is unchanged|is unchanged|unchanged\.|is correct as written|"
                    r"no changes (are )?needed)", re.I)
def claim(reply):
    return not NOPROB.search(reply[:3000])
def table(name, rows, wrong_before):
    c = Counter()
    for r in rows:
        x = r["critique"][0]
        c[(("claims problem" if claim(x["reply"]) else "says fine"), ("changed" if x["changed"] else "same"),
           ("passes" if x["ok"] else "fails"))] += 1
    print(f"\n{name} (n={len(rows)}, first attempt {'wrong' if wrong_before else 'right'})")
    for k, v in sorted(c.items()): print("  ", k, v)
main = [json.loads(l) for l in open("runs/main.jsonl")]
calls = {}
for l in open("runs/calls-main.jsonl"):
    c = json.loads(l)
    if c["note"] == "critique r1": calls[c["tag"]] = c["reply"]
for r in main: r["critique"][0]["reply"] = calls.get(f"{r['task']}#{r['rep']}", "")
table("main, attempt 1 right", [r for r in main if r["single"]["ok"]], False)
table("main, attempt 1 wrong", [r for r in main if not r["single"]["ok"]], True)
table("natural pool, Haiku", [json.loads(l) for l in open("runs/repair-nat.jsonl")], True)
table("natural pool, Sonnet", [json.loads(l) for l in open("runs/repair-nat-sonnet.jsonl")], True)
table("planted bugs, Haiku", [json.loads(l) for l in open("runs/repair-mut.jsonl")], True)
