import json
def slice_docs(path, mb):
    """First documents of a jsonl file, up to `mb` megabytes of UTF-8 text."""
    out, n = [], 0
    with open(path) as f:
        for line in f:
            try: t = json.loads(line)["text"]
            except Exception: break
            out.append(t); n += len(t.encode())
            if n >= mb * 1e6: break
    return out, n
