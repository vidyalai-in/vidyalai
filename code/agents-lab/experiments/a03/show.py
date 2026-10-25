"""Print a transcript compactly: python show.py <path> [maxchars]"""
import json, sys
n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e["event"] in ("plan", "replan"):
        print(f"=== {e['event'].upper()} ===\n{e['plan'][:n*3]}")
    elif e["event"] == "step":
        print(f"--- step {e['n']} out {e['output_tokens']} thinking {e.get('thinking_tokens')} {e['stop_reason']}")
        for b in e["content"]:
            t = b.get("type")
            body = b.get("text") or b.get("thinking") or json.dumps(b.get("input"))
            print(f"  [{t}] {body[:n]}")
    elif e["event"] == "tool_results":
        for r in e["results"]:
            print("    ->", r["content"][:300].replace("\n", " | "))
    elif e["event"] == "result":
        print("RESULT", e["stop"], e["answer"][-200:])
