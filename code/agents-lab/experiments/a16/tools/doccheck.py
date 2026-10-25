"""Run the doctest examples of one .rst file against the semver library. Prints failures as JSON lines.
Usage: python doccheck.py <file.rst>   (run inside the sandbox; semver importable via PYTHONPATH)"""
import doctest, json, sys, warnings
warnings.simplefilter("ignore")
import semver
from semver.version import Version

path = sys.argv[1]
text = open(path).read()
parser = doctest.DocTestParser()
test = parser.get_doctest(text, {"semver": semver, "Version": Version}, path, path, 0)
fails = []

class R(doctest.DocTestRunner):
    def report_failure(self, out, test, ex, got):
        if got.startswith("Traceback"):
            got = "Traceback (most recent call last):\n...\n" + got.strip().splitlines()[-1]
        fails.append({"line": ex.lineno + 1, "source": ex.source.strip(), "expected": ex.want.strip(), "got": got.strip()[:300]})
    def report_unexpected_exception(self, out, test, ex, exc_info):
        e = exc_info[1]
        fails.append({"line": ex.lineno + 1, "source": ex.source.strip(), "expected": ex.want.strip()[:300],
                      "got": f"{type(e).__name__}: {e}"[:300]})

r = R(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
r.run(test, out=lambda s: None)
print(json.dumps({"examples": len(test.examples), "failures": fails}))
