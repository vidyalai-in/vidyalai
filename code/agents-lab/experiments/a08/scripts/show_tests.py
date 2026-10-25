import ast, json, sys
from pathlib import Path
A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
want = dict(a.split(":") for a in sys.argv[1:])  # repo:commitprefix
for repo, pre in want.items():
    for l in (A08/"candidates"/f"{repo}.jsonl").read_text().splitlines():
        c = json.loads(l)
        if not c["commit"].startswith(pre): continue
        print(f"######## {repo} {pre} {c['subject'][:70]}")
        seen = set()
        for t in c["f2p"]:
            path, *parts = t.split("[")[0].split("::")
            key = (path, tuple(parts))
            if key in seen: continue
            seen.add(key)
            src = (A08/"repos"/repo/path).read_text(); tree = ast.parse(src)
            def find(body, parts):
                for n in body:
                    if getattr(n, "name", None) == parts[0]:
                        return n if len(parts) == 1 else find(n.body, parts[1:])
            n = find(tree.body, parts)
            if n:
                start = min([d.lineno for d in n.decorator_list] + [n.lineno])
                print("\n".join(src.splitlines()[start-1:n.end_lineno])[:1800])
