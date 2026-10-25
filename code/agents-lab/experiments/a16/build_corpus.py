"""Copy python-semver's docs and source into corpus/clean and corpus/seeded, then apply the seeds."""
import shutil
from pathlib import Path
from seeds import SEEDS

A16 = Path(__file__).parent
SRC = A16 / "semver-src"
import sys
for name in ([] if sys.argv[1:] == ["hard"] else ["clean", "seeded"]):
    d = A16 / "corpus" / name
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC / "docs", d / "docs", ignore=shutil.ignore_patterns("_static", "_templates", "inventories", "*.py", "Makefile", "make.bat"))
    shutil.copytree(SRC / "src" / "semver", d / "src" / "semver", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(SRC / "CHANGELOG.rst", d / "CHANGELOG.rst")
for s in ([] if sys.argv[1:] == ["hard"] else SEEDS):
    p = A16 / "corpus" / "seeded" / "docs" / s["file"]
    lines = p.read_text().split("\n")
    line = lines[s["line"] - 1]
    assert line.count(s["old"]) == 1, (s["id"], line)
    lines[s["line"] - 1] = line.replace(s["old"], s["new"])
    p.write_text("\n".join(lines))
print("seeded", len(SEEDS))

from seeds import HARD
d = A16 / "corpus" / "hard"
shutil.rmtree(d, ignore_errors=True)
shutil.copytree(A16 / "corpus" / "clean", d)
for s in HARD:
    p = d / "docs" / s["file"]
    lines = p.read_text().split("\n")
    assert lines[s["line"] - 1].count(s["old"]) == 1, (s["id"], lines[s["line"] - 1])
    lines[s["line"] - 1] = lines[s["line"] - 1].replace(s["old"], s["new"])
    p.write_text("\n".join(lines))
print("hard", len(HARD))
