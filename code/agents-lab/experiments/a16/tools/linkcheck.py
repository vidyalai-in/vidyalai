"""Check the cross-references of one .rst file against the docs tree and the semver library.
Usage: python linkcheck.py <docs_root> <file.rst>. Prints JSON {"links": n, "broken": [...]}.
Roles checked: :ref: (labels defined with '.. _label:' anywhere in the tree), :doc: (document paths),
and the Python roles :meth: :func: :class: :attr: :exc: :data: :const: :mod: (resolved by import)."""
import builtins, difflib, importlib, json, re, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")

root, path = Path(sys.argv[1]), Path(sys.argv[2])
labels = set()
for f in root.rglob("*.rst"):
    labels |= {m.lower() for m in re.findall(r"^\.\. _([^:]+):\s*$", f.read_text(), re.M)}
docs = {str(f.relative_to(root).with_suffix("")) for f in root.rglob("*.rst")}
ROLE = re.compile(r":(ref|doc|meth|func|class|attr|exc|data|const|mod):`([^`]+)`")
PREFIXES = ["", "semver.", "semver.version.", "semver.version.Version."]


def target(body):
    m = re.search(r"<([^>]+)>\s*$", body)
    return (m.group(1) if m else body).strip()


def suggest_py(name):
    """Close attribute names on the deepest object that does resolve (for a fix suggestion)."""
    name = name.lstrip("~.").rstrip("()")
    if ":" in name:
        return []
    for pre in PREFIXES:
        parts = (pre + name).split(".")
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except Exception:
                continue
            j = i
            for a in parts[i:-1]:
                if not hasattr(obj, a):
                    break
                obj, j = getattr(obj, a), j + 1
            if j == len(parts) - 1:
                close = difflib.get_close_matches(parts[-1], [d for d in dir(obj) if not d.startswith("_") or d.startswith("__")], 3, 0.6)
                return [".".join(parts[:-1] + [c]) for c in close]
            break
    return []


def resolve_py(name):
    name = name.lstrip("~.").rstrip("()")
    if ":" in name:  # intersphinx: python:X is checked against the standard library, others skipped
        inv, name = name.split(":", 1)
        if inv != "python":
            return True
        if hasattr(builtins, name.split(".")[0]):
            return True
        try:
            importlib.import_module(name.split(".")[0])
            return True
        except Exception:
            return False
    for pre in PREFIXES:
        parts = (pre + name).split(".")
        if len(parts) == 1 and hasattr(builtins, parts[0]):
            return True
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except Exception:
                continue
            try:
                for a in parts[i:]:
                    obj = getattr(obj, a)
                return True
            except AttributeError:
                break
    return False


broken, n = [], 0
for lineno, line in enumerate(path.read_text().splitlines(), 1):
    for role, body in ROLE.findall(line):
        n += 1
        t = target(body)
        if role == "ref":
            ok = t.lower() in labels
        elif role == "doc":
            base = Path("/") if t.startswith("/") else Path("/") / path.relative_to(root).parent
            ok = str((base / t.lstrip("/")).resolve())[1:] in docs
        else:
            ok = resolve_py(t)
        if not ok:
            if role == "ref":
                close = difflib.get_close_matches(t.lower(), sorted(labels), 3, 0.6)
            elif role == "doc":
                close = difflib.get_close_matches(t.lstrip("/"), sorted(docs), 3, 0.6)
            else:
                close = suggest_py(t)
            broken.append({"line": lineno, "role": role, "target": t, "close_matches": close})
print(json.dumps({"links": n, "broken": broken}))
