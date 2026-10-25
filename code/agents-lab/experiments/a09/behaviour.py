"""Behaviour metrics from transcripts: actions per turn, observations used, refused members, recoveries."""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

pat = sys.argv[1] if len(sys.argv) > 1 else "*-haiku-medium-r*"
OBS = {"screenshot", "zoom", "read_page", "find", "get_page_text"}
agg = defaultdict(Counter)
for f in glob.glob(f"/home/amit/.cache/vidyalai-scratch/a09/runs/{pat}.jsonl"):
    name = f.rsplit("/", 1)[1]
    arm = name.split("-haiku")[0].split("-", 1)[1]
    turns = []
    for l in open(f):
        d = json.loads(l)
        if d["event"] == "step":
            calls = [b for b in d["content"] if b["type"] == "tool_use"]
            if calls:
                turns.append([b["name"] for b in calls])
            txt = " ".join(b.get("thinking", "") + b.get("text", "") for b in d["content"] if b["type"] in ("thinking", "text"))
            if re.search(r"didn't (register|take|stick|work)|did not register|re-enter|retry|missed|wrong (field|element|place)|instead of", txt, re.I):
                agg[arm]["recovery_msgs"] += 1
        if d["event"] == "tool_results":
            for r in d["results"]:
                c = str(r.get("content"))
                if "not available in this browser" in c:
                    agg[arm]["disabled_member_calls"] += 1
                if "stale or not found" in c:
                    agg[arm]["stale_ref"] += 1
                if "Not executed" in c:
                    agg[arm]["halted"] += 1
                if "not visible" in c:
                    agg[arm]["not_visible"] += 1
    a = agg[arm]
    a["runs"] += 1
    a["turns"] += len(turns)
    for t in turns:
        a["actions"] += len(t)
        a["obs"] += sum(x in OBS for x in t)
        a["multi"] += len(t) >= 3
        for x in t:
            a["m:" + x] += 1
for arm, a in agg.items():
    print(arm, f"runs {a['runs']} turns/run {a['turns']/a['runs']:.1f} actions/turn {a['actions']/a['turns']:.2f} "
               f"turns with >=3 calls {100*a['multi']/a['turns']:.0f}% observations {100*a['obs']/a['actions']:.0f}% of calls "
               f"disabled-member calls {a['disabled_member_calls']} stale {a['stale_ref']} halted {a['halted']} not_visible {a['not_visible']} recovery msgs {a['recovery_msgs']}")
    print("   ", ", ".join(f"{k[2:]} {v}" for k, v in a.most_common() if k.startswith("m:"))[:600])
