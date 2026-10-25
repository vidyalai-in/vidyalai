"""Collect every cited claim from the transcripts (final runs only) into cites.jsonl."""
import json, collections
from pathlib import Path
out = []
for f in sorted(Path("runs").glob("*.jsonl")):
    for i, l in enumerate(f.read_text().splitlines()):
        r = json.loads(l)
        if r["model"] == "claude-sonnet-5-5" and r.get("client_calls") is None and "Rust release" in r["question"]:
            continue  # the Sonnet s02 attempt that stopped on a nonexistent tool (no citations anyway)
        docs = [b["content"]["url"] for b in r["content"]
                if b["type"] == "web_fetch_tool_result" and b["content"].get("type") == "web_fetch_result"]
        for b in r["content"]:
            for c in b.get("citations") or []:
                url = c.get("url") or (docs[c["document_index"]] if c.get("document_index") is not None and c["document_index"] < len(docs) else None)
                out.append(dict(run=f"{f.stem}#{i}", arm=r["arm"], model=r["model"], question=r["question"],
                                claim=b["text"], ctype=c["type"], url=url, cited_text=c["cited_text"]))
Path("cites.jsonl").write_text("\n".join(json.dumps(x) for x in out) + "\n")
c = collections.Counter((x["arm"], x["model"], x["ctype"]) for x in out)
for k, v in sorted(c.items()): print(k, v)
runs = collections.Counter((x["arm"], x["model"]) for x in out)
print("distinct urls", len({x["url"] for x in out}))
