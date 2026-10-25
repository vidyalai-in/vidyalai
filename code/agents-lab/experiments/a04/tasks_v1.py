"""40 small Python function-writing tasks with unit tests, for Agentic AI Part 4.

Each task has a spec (shown to the model), a reference solution (never shown), and test inputs.
Expected outputs are computed from the reference. The first `n_public` tests form the public
suite (what the verifier tool runs); the full list is the hidden suite used for grading.
"""

TASKS = []


def task(name, spec, ref, tests, n_public):
    TASKS.append(dict(id=name, spec=spec.strip(), ref=ref.strip(), tests=tests, n_public=n_public))


task("human_size", '''
def human_size(n: int) -> str:
    """Format a byte count with binary units.
    - |n| < 1024: the integer followed by " B", e.g. "1023 B", "0 B".
    - Otherwise divide by 1024 until the value is below 1024, using units KiB, MiB, GiB, TiB
      (never beyond TiB: larger values stay in TiB, e.g. "2048.0 TiB").
    - Show exactly one decimal, rounding half up (1.25 -> "1.3", 1.24 -> "1.2").
    - If rounding gives 1024.0 and a larger unit exists, use the next unit instead ("1.0 MiB").
    - Negative numbers get a leading "-" and are otherwise formatted like their absolute value.
    Example: human_size(1536) == "1.5 KiB"
    """
''', '''
from decimal import Decimal, ROUND_HALF_UP
def human_size(n):
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1024:
        return f"{sign}{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    v = Decimal(n) / 1024
    i = 0
    while v >= 1024 and i < 3:
        v /= 1024
        i += 1
    r = v.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if r >= 1024 and i < 3:
        i += 1
        r = (v / 1024).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{sign}{r} {units[i]}"
''', ["(1536)", "(1023)", "(0)", "(1048576)", "(1280)", "(1048575)", "(-2048)",
      "(1024**5 * 2)", "(1126)", "(3 * 1024**3 + 51 * 1024**2)", "(1024**4 - 1)"], 4)

task("parse_duration", '''
def parse_duration(s: str) -> int:
    """Parse a duration like "1h30m" or "2d 4h" into a number of seconds.
    - Components are a non-negative integer followed by a unit: d (86400 s), h, m, s.
    - Units are lowercase, each appears at most once, and they must be in the order d, h, m, s.
    - Components may be separated by single spaces; no other whitespace is allowed anywhere
      (no leading or trailing spaces, no space between number and unit).
    - Anything else (empty string, decimals, unknown units, repeated or out-of-order units)
      raises ValueError.
    Example: parse_duration("1h30m") == 5400
    """
''', '''
import re
def parse_duration(s):
    m = re.fullmatch(r"(?:(\\d+)d)?(?:(?<=d) ?)?(?:(\\d+)h)?(?:(?<=h) ?)?(?:(\\d+)m)?(?:(?<=m) ?)?(?:(\\d+)s)?", s)
    if not s or not m or s.endswith(" "):
        raise ValueError(s)
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
''', ['("1h30m")', '("45s")', '("2d 4h")', '("")', '("0s")', '("1h1h")', '("30m1h")', '("1.5h")',
      '(" 1h")', '("1H")', '("1d 2h 3m 4s")', '("1h  30m")', '("1 h")', '("10m ")', '("100s")'], 5)

task("compare_versions", '''
def compare_versions(a: str, b: str) -> int:
    """Compare two version strings; return -1 if a < b, 0 if equal, 1 if a > b.
    - A version is dot-separated non-negative integers, optionally followed by "-" and a
      pre-release, optionally followed by "+" and build metadata.
    - Core numbers compare numerically; missing trailing numbers count as 0 ("1.2" == "1.2.0").
    - Build metadata (after "+") is ignored entirely.
    - With equal cores, a version with a pre-release is lower than one without.
    - Two pre-releases compare by their dot-separated identifiers, left to right: identifiers
      of only digits compare numerically; others compare as ASCII strings; a numeric identifier
      is lower than a non-numeric one; if all compared identifiers are equal, the one with fewer
      identifiers is lower.
    Example: compare_versions("1.10", "1.9") == 1
    """
''', '''
def compare_versions(a, b):
    def split(v):
        v = v.split("+", 1)[0]
        core, _, pre = v.partition("-")
        nums = [int(x) for x in core.split(".")]
        return nums, (pre.split(".") if pre else None)
    (ca, pa), (cb, pb) = split(a), split(b)
    n = max(len(ca), len(cb))
    ca += [0] * (n - len(ca)); cb += [0] * (n - len(cb))
    if ca != cb:
        return -1 if ca < cb else 1
    if pa is None and pb is None: return 0
    if pa is None: return 1
    if pb is None: return -1
    for x, y in zip(pa, pb):
        if x == y: continue
        xd, yd = x.isdigit(), y.isdigit()
        if xd and yd: return -1 if int(x) < int(y) else 1
        if xd != yd: return -1 if xd else 1
        return -1 if x < y else 1
    if len(pa) == len(pb): return 0
    return -1 if len(pa) < len(pb) else 1
''', ['("1.10", "1.9")', '("1.2", "1.2.0")', '("1.0.0-alpha", "1.0.0")', '("2.0.0", "10.0.0")',
      '("1.0.0+build.5", "1.0.0")', '("1.0.0-alpha.1", "1.0.0-alpha")', '("1.0.0-alpha.beta", "1.0.0-alpha.1")',
      '("1.0.0-rc.10", "1.0.0-rc.9")', '("1.0.0-Beta", "1.0.0-alpha")', '("1.0.0-alpha+x", "1.0.0-alpha+y")',
      '("1.0.0-2", "1.0.0-10")', '("1", "1.0.0.0")', '("1.0.0-beta.11", "1.0.0-beta.2")'], 5)

task("roman_to_int", '''
def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral to an integer, or return None if it is not valid.
    Valid numerals are exactly the standard (canonical) forms of 1..3999 in uppercase:
    subtractive pairs only IV, IX, XL, XC, CD, CM; no symbol repeated more than three times
    in a row; V, L, D never repeated. Lowercase, empty strings and non-canonical forms such as
    "IIII", "IC", "VX" or "XIIX" return None.
    Example: roman_to_int("MCMXCIV") == 1994
    """
''', '''
def roman_to_int(s):
    vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    table = {}
    for n in range(1, 4000):
        out, x = "", n
        for v, r in vals:
            while x >= v:
                out += r; x -= v
        table[out] = n
    return table.get(s)
''', ['("MCMXCIV")', '("IIII")', '("XLII")', '("")', '("IC")', '("mmxx")', '("MMMCMXCIX")',
      '("VX")', '("XIIX")', '("MMMM")', '("CDXLIV")', '("IIX")', '("DD")', '("XCIX")'], 4)

task("int_to_words", '''
def int_to_words(n: int) -> str:
    """Spell out an integer in English words, for |n| < 10**12.
    - Lowercase words separated by single spaces; numbers 21-99 that are not multiples of ten
      are hyphenated ("forty-two"); no "and".
    - Scale words: hundred, thousand, million, billion; omit zero groups
      (1000001 -> "one million one").
    - 0 -> "zero"; negative numbers start with "minus ".
    Example: int_to_words(123) == "one hundred twenty-three"
    """
''', '''
def int_to_words(n):
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    def lt1000(x):
        parts = []
        if x >= 100:
            parts.append(ones[x // 100] + " hundred"); x %= 100
        if x >= 20:
            parts.append(tens[x // 10] + ("-" + ones[x % 10] if x % 10 else ""))
        elif x:
            parts.append(ones[x])
        return " ".join(parts)
    if n == 0: return "zero"
    if n < 0: return "minus " + int_to_words(-n)
    out = []
    for scale, word in [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (1, "")]:
        g = n // scale; n %= scale
        if g:
            out.append(lt1000(g) + (" " + word if word else ""))
    return " ".join(out)
''', ["(123)", "(0)", "(42)", "(1000001)", "(-15)", "(999999999999)", "(100)", "(1000000000)",
      "(110)", "(2000020)", "(-90)", "(700000000017)"], 4)

task("wrap_text", '''
def wrap_text(text: str, width: int) -> list[str]:
    """Greedy word wrap.
    - Words are maximal runs of non-whitespace; any run of whitespace (including newlines)
      separates words. Lines join words with a single space and never exceed `width`.
    - Fill each line with as many words as fit, then start a new line.
    - A word longer than `width` starts on a new line (unless the current line is empty) and is
      cut into pieces of exactly `width` characters, each on its own line; its last piece can be
      followed by further words on the same line if they fit.
    - Text with no words returns [].
    Example: wrap_text("the quick brown fox", 10) == ["the quick", "brown fox"]
    """
''', '''
def wrap_text(text, width):
    lines, cur = [], ""
    for w in text.split():
        if len(w) > width:
            if cur:
                lines.append(cur); cur = ""
            while len(w) > width:
                lines.append(w[:width]); w = w[width:]
            cur = w
            continue
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines
''', ['("the quick brown fox", 10)', '("", 5)', '("a  b\\n\\nc", 3)', '("abcdefghij", 4)',
      '("hi abcdefghij yo", 4)', '("ab abcdefgh c", 4)', '("   ", 3)', '("one two three", 3)',
      '("x abcdef", 3)', '("aaaa bbbb", 4)'], 4)

task("merge_intervals", '''
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge half-open intervals [start, end).
    - Intervals that overlap or touch (one's end equals another's start) are merged.
    - Empty intervals (start == end) are dropped; if any interval has start > end, raise ValueError.
    - Return the merged intervals sorted by start, as lists [start, end]. Do not modify the input.
    Example: merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
    """
''', '''
def merge_intervals(intervals):
    if any(s > e for s, e in intervals):
        raise ValueError("bad interval")
    out = []
    for s, e in sorted([list(x) for x in intervals if x[0] < x[1]]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''', ["([[1, 3], [2, 6], [8, 10]])", "([])", "([[5, 7], [1, 2]])", "([[1, 3], [3, 5]])",
      "([[2, 2], [1, 1]])", "([[4, 2]])", "([[1, 10], [2, 3], [4, 5]])", "([[3, 3], [1, 3], [3, 4]])",
      "([[0, 1], [5, 5], [1, 2], [7, 9]])"], 4)

task("csv_split", '''
def csv_split(line: str) -> list[str]:
    """Split one CSV line into fields.
    - Fields are separated by commas. A field that starts with a double quote is quoted: inside
      it, commas are literal and two double quotes ("") stand for one double quote; the field
      ends at the closing quote, which must be followed by a comma or the end of the line
      (otherwise raise ValueError). An unterminated quoted field raises ValueError.
    - In an unquoted field every character, including spaces and double quotes, is literal.
    - "" (empty line) -> [""]; a trailing comma gives a trailing empty field.
    Example: csv_split('a,"b,c",d') == ["a", "b,c", "d"]
    """
''', '''
def csv_split(line):
    fields, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1; buf = []
            while True:
                if i >= n: raise ValueError("unterminated")
                c = line[i]
                if c == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"'); i += 2; continue
                    i += 1; break
                buf.append(c); i += 1
            fields.append("".join(buf))
            if i == n: return fields
            if line[i] != ",": raise ValueError("junk after quote")
            i += 1
        else:
            j = line.find(",", i)
            if j == -1:
                fields.append(line[i:]); return fields
            fields.append(line[i:j]); i = j + 1
''', ["('a,\"b,c\",d')", "('')", "('a,')", "('\"x\"\"y\"')", "('\"abc')", "('a b, c')",
      "('ab\"c\",d')", "('\"a\"b,c')", "(',,')", "('\"\"')", "('\"a,b\",')", "('x,\"\"\"\"')"], 4)

task("is_valid_ipv4", '''
def is_valid_ipv4(s: str) -> bool:
    """Return True if s is a dotted-decimal IPv4 address: exactly four parts separated by dots,
    each part one to three ASCII digits (0-9) with value 0-255 and no leading zeros
    ("0" is fine, "01" is not). No signs, spaces or other characters anywhere.
    Example: is_valid_ipv4("192.168.1.1") is True
    """
''', '''
def is_valid_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4: return False
    for p in parts:
        if not p or len(p) > 3 or any(c not in "0123456789" for c in p): return False
        if p != "0" and p[0] == "0": return False
        if int(p) > 255: return False
    return True
''', ['("192.168.1.1")', '("256.1.1.1")', '("1.2.3")', '("01.2.3.4")', '("0.0.0.0")', '("1.2.3.4 ")',
      '("1..3.4")', '("+1.2.3.4")', '("1.2.3.\\u0664")', '("255.255.255.255")', '("1.2.3.4.5")',
      '("1.2.3.0000")', '("1.2.3.00")'], 4)

task("luhn_valid", '''
def luhn_valid(s: str) -> bool:
    """Check a number with the Luhn algorithm.
    - Spaces are ignored; any other non-digit character (only ASCII 0-9 count as digits) makes
      it invalid.
    - Strings with fewer than two digits are invalid.
    - Starting from the rightmost digit, double every second digit (the second-to-last, the
      fourth-to-last, ...); if doubling gives more than 9, subtract 9. Valid if the sum of all
      digits is divisible by 10.
    Example: luhn_valid("4539 3195 0343 6467") is True
    """
''', '''
def luhn_valid(s):
    s = s.replace(" ", "")
    if len(s) < 2 or any(c not in "0123456789" for c in s): return False
    total = 0
    for i, c in enumerate(reversed(s)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        total += d
    return total % 10 == 0
''', ['("4539 3195 0343 6467")', '("8273 1232 7352 0569")', '("0")', '("00")', '(" 0 ")', '("059a")',
      '("0 0")', '("091")', '("1234-5678")', '("79927398713")', '("  ")', '("4111111111111111")'], 4)

task("snake_case", '''
def snake_case(name: str) -> str:
    """Convert a camelCase or PascalCase identifier (ASCII letters, digits, underscores) to
    snake_case.
    - Insert "_" before an uppercase letter that follows a lowercase letter or a digit.
    - In a run of uppercase letters (an acronym), insert "_" before the last uppercase letter
      of the run if it is followed by a lowercase letter ("HTTPServer" -> "http_server").
    - Digits never start a new word ("utf8Encode" -> "utf8_encode", "Version2" -> "version2").
    - Existing underscores are kept, and never doubled: "get_HTTPCode" -> "get_http_code".
    - The result is lowercase.
    Example: snake_case("getHTTPResponseCode") == "get_http_response_code"
    """
''', '''
def snake_case(name):
    out = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                if out and out[-1] != "_":
                    out.append("_")
        out.append(c.lower())
    return "".join(out)
''', ['("getHTTPResponseCode")', '("HTTPServer")', '("already_snake")', '("utf8Encode")', '("Version2")',
      '("get_HTTPCode")', '("A")', '("ABC")', '("x_Y")', '("simpleXMLParser2Go")', '("_Private")',
      '("IOError")'], 4)

task("fraction_to_decimal", '''
def fraction_to_decimal(num: int, den: int) -> str:
    """Write num/den as a decimal string, putting a repeating part in parentheses.
    - Exact results have no trailing zeros and no trailing "." ("2", "0.5", "-0.25").
    - A repeating part is the shortest cycle, starting as early as possible: 1/3 -> "0.(3)",
      1/6 -> "0.1(6)", 1/7 -> "0.(142857)".
    - A negative result has a single leading "-"; zero is always "0".
    - den == 0 raises ZeroDivisionError.
    Example: fraction_to_decimal(1, 6) == "0.1(6)"
    """
''', '''
def fraction_to_decimal(num, den):
    if den == 0: raise ZeroDivisionError
    if num == 0: return "0"
    sign = "-" if (num < 0) != (den < 0) else ""
    num, den = abs(num), abs(den)
    q, r = divmod(num, den)
    if r == 0: return f"{sign}{q}"
    digits, seen = [], {}
    while r and r not in seen:
        seen[r] = len(digits)
        r *= 10
        digits.append(str(r // den)); r %= den
    if r:
        i = seen[r]
        return f"{sign}{q}." + "".join(digits[:i]) + "(" + "".join(digits[i:]) + ")"
    return f"{sign}{q}." + "".join(digits)
''', ["(1, 6)", "(1, 3)", "(4, 2)", "(-1, 4)", "(1, 0)", "(0, -5)", "(-1, -3)", "(22, 7)",
      "(1, -3)", "(1, 7)", "(5, 12)", "(-50, 8)", "(1, 333)"], 4)

task("simplify_path", '''
def simplify_path(path: str) -> str:
    """Normalize a Unix-style path.
    - Repeated slashes count as one; "." components are removed; a ".." removes the previous
      component.
    - Absolute paths (starting with "/") stay absolute, and ".." at the root is dropped
      ("/../a" -> "/a"). The root alone is "/".
    - Relative paths keep any leading ".." that cannot be resolved ("a/../../b" -> "../b").
      An empty result for a relative path (or an empty input) is ".".
    - The result has no trailing slash (except "/").
    Example: simplify_path("/a/./b/../../c/") == "/c"
    """
''', '''
def simplify_path(path):
    absolute = path.startswith("/")
    stack = []
    for part in path.split("/"):
        if part in ("", "."): continue
        if part == "..":
            if stack and stack[-1] != "..": stack.pop()
            elif not absolute: stack.append("..")
        else:
            stack.append(part)
    if absolute: return "/" + "/".join(stack)
    return "/".join(stack) or "."
''', ['("/a/./b/../../c/")', '("/../")', '("a/../../b")', '("")', '("//a//b/")', '("./")',
      '("../../x/..")', '("a/b/..")', '("/a/b/../../..")', '("a/./b/./c/../d")', '("..")'], 4)

task("split_bill", '''
def split_bill(total_cents: int, weights: list[int]) -> list[int]:
    """Split total_cents (>= 0) into shares proportional to non-negative integer weights.
    - Each share starts as floor(total_cents * w / sum(weights)).
    - The leftover cents are given out one at a time to the shares with the largest fractional
      remainders; ties go to the lower index.
    - The shares always sum to total_cents. If the weights sum to 0, raise ValueError.
    Example: split_bill(100, [1, 1, 1]) == [34, 33, 33]
    """
''', '''
def split_bill(total_cents, weights):
    s = sum(weights)
    if s == 0: raise ValueError("zero weights")
    base = [total_cents * w // s for w in weights]
    rem = [total_cents * w % s for w in weights]
    left = total_cents - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-rem[i], i))
    for i in order[:left]:
        base[i] += 1
    return base
''', ["(100, [1, 1, 1])", "(10, [0, 0])", "(7, [1, 2])", "(0, [3, 4])", "(101, [1, 2, 3, 4])",
      "(5, [0, 1, 0, 1])", "(99, [3, 3, 3])", "(1000, [7, 11, 13])", "(1, [1, 1, 1, 1])", "(10, [5])"], 4)

task("next_permutation", '''
def next_permutation(seq: list) -> list:
    """Return the next permutation of seq in lexicographic order, as a new list.
    Values may repeat (permutations are of the multiset, so [1, 1, 2] -> [1, 2, 1]).
    If seq is already the last permutation (non-increasing), return it sorted ascending.
    Do not modify the input. [] -> [].
    Example: next_permutation([1, 2, 3]) == [1, 3, 2]
    """
''', '''
def next_permutation(seq):
    a = list(seq)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]: i -= 1
    if i < 0: return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]: j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a
''', ["([1, 2, 3])", "([3, 2, 1])", "([1, 1, 2])", "([])", "([1, 3, 2])", "([2, 2, 1, 1])",
      "([1, 5, 1])", "([7])", "(['a', 'c', 'b'])", "([1, 2, 2, 1])"], 4)

task("glob_match", '''
def glob_match(pattern: str, text: str) -> bool:
    """Match text against a glob pattern; the whole text must match.
    - "?" matches exactly one character other than "/".
    - "*" matches any run of characters (possibly empty) that contains no "/".
    - "**" matches any run of characters, including "/".
    - "[abc]" matches one listed character; "[a-z]" a range; "[!abc]" any character other than
      "/" not listed. A class never matches "/".
    - Every other character matches itself.
    Example: glob_match("src/*.py", "src/main.py") is True
    """
''', '''
import re
def glob_match(pattern, text):
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*"); i += 2
        elif c == "*":
            out.append("[^/]*"); i += 1
        elif c == "?":
            out.append("[^/]"); i += 1
        elif c == "[":
            j = pattern.index("]", i + 1)
            body = pattern[i + 1:j]
            if body.startswith("!"):
                out.append("[^/" + body[1:].replace("\\\\", "\\\\\\\\") + "]")
            else:
                out.append("(?!/)[" + body.replace("\\\\", "\\\\\\\\") + "]")
            i = j + 1
        else:
            out.append(re.escape(c)); i += 1
    return re.fullmatch("".join(out), text, re.S) is not None
''', ['("src/*.py", "src/main.py")', '("*.py", "a/b.py")', '("**.py", "a/b.py")', '("?", "/")',
      '("a?c", "abc")', '("[a-c]x", "bx")', '("[!a-c]x", "dx")', '("[!a-c]x", "/x")', '("a/**/b", "a/x/y/b")',
      '("*", "")', '("[ab]", "c")', '("a*b*c", "aXXbYYc")', '("**", "")', '("a/*", "a/b/c")'], 5)

task("sliding_median", '''
def sliding_median(nums: list[int], k: int) -> list[float]:
    """Return the median of every window of k consecutive numbers, left to right, as floats.
    For even k the median is the mean of the two middle values. If k > len(nums) return [].
    If k <= 0 raise ValueError.
    Example: sliding_median([1, 3, -1, -3, 5], 3) == [1.0, -1.0, -1.0]
    """
''', '''
def sliding_median(nums, k):
    if k <= 0: raise ValueError
    out = []
    for i in range(len(nums) - k + 1):
        w = sorted(nums[i:i + k])
        out.append(float(w[k // 2]) if k % 2 else (w[k // 2 - 1] + w[k // 2]) / 2)
    return out
''', ["([1, 3, -1, -3, 5], 3)", "([1, 2, 3, 4], 2)", "([1, 2], 3)", "([5], 1)", "([1, 2], 0)",
      "([4, 1, 7, 2, 8, 3], 4)", "([2, 2, 2], 2)", "([1, 4, 2, 3], 4)"], 4)

task("nth_weekday", '''
def nth_weekday(year: int, month: int, weekday: int, n: int) -> int | None:
    """Return the day of the month of the n-th given weekday (0 = Monday ... 6 = Sunday).
    n = 1..5 counts from the start of the month; n = -1 means the last such weekday.
    Return None if that weekday does not occur n times in the month.
    Example: nth_weekday(2026, 11, 3, 4) == 26   # 4th Thursday of November 2026
    """
''', '''
import calendar
def nth_weekday(year, month, weekday, n):
    days = [d for d in range(1, calendar.monthrange(year, month)[1] + 1)
            if calendar.weekday(year, month, d) == weekday]
    if n == -1: return days[-1]
    return days[n - 1] if 1 <= n <= len(days) else None
''', ["(2026, 11, 3, 4)", "(2026, 2, 0, 5)", "(2024, 2, 3, 5)", "(2026, 5, 0, -1)", "(2026, 1, 3, 1)",
      "(2026, 3, 6, 5)", "(2025, 12, 2, -1)", "(2023, 2, 1, 4)"], 3)

task("business_days", '''
def business_days(start: str, end: str, holidays: list[str]) -> int:
    """Count business days (Monday-Friday, not in holidays) in the half-open range [start, end).
    Dates are "YYYY-MM-DD" strings. Holidays that fall on weekends change nothing; duplicate
    holidays count once. If end is before start, return the negative of the count for
    [end, start). start == end gives 0.
    Example: business_days("2026-10-19", "2026-10-26", []) == 5
    """
''', '''
from datetime import date, timedelta
def business_days(start, end, holidays):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    sign = 1
    if e < s: s, e, sign = e, s, -1
    hol = {date.fromisoformat(h) for h in holidays}
    n, d = 0, s
    while d < e:
        if d.weekday() < 5 and d not in hol: n += 1
        d += timedelta(days=1)
    return sign * n
''', ['("2026-10-19", "2026-10-26", [])', '("2026-10-19", "2026-10-19", [])', '("2026-10-24", "2026-10-26", [])',
      '("2026-10-26", "2026-10-19", [])', '("2026-12-21", "2027-01-04", ["2026-12-25", "2027-01-01"])',
      '("2026-10-19", "2026-10-26", ["2026-10-24", "2026-10-20", "2026-10-20"])',
      '("2026-10-26", "2026-10-19", ["2026-10-21"])', '("2024-02-26", "2024-03-04", [])'], 3)

task("rle_decode", '''
def rle_decode(s: str) -> str:
    """Decode a run-length string where an optional decimal count precedes each character.
    - "3a2b" -> "aaabb"; a character without a count appears once ("a3b" -> "abbb").
    - Counts may have several digits ("12x") and may be 0 (the character is dropped).
    - A backslash makes the next character literal, so digits and backslashes can be encoded:
      "2\\\\3" (the 3-character string 2, backslash, 3) -> "33".
    - A count or backslash with no character after it raises ValueError.
    Example: rle_decode("3a2b") == "aaabb"
    """
''', '''
def rle_decode(s):
    out, i, n = [], 0, len(s)
    while i < n:
        j = i
        while j < n and s[j].isdigit() and s[j] in "0123456789": j += 1
        count = int(s[i:j]) if j > i else 1
        if j >= n: raise ValueError("dangling count")
        if s[j] == "\\\\":
            if j + 1 >= n: raise ValueError("dangling escape")
            ch, i = s[j + 1], j + 2
        else:
            ch, i = s[j], j + 1
        out.append(ch * count)
    return "".join(out)
''', ['("3a2b")', '("a3b")', '("12x")', '("2\\\\3")', '("0ab")', '("3")', '("a\\\\")',
      '("\\\\\\\\2\\\\\\\\")', '("")', '("10\\\\5z")', '("1a1b1c")'], 4)

task("balanced", '''
def balanced(s: str) -> bool:
    """Return True if the brackets (), [] and {} in s are balanced and properly nested.
    - Characters inside single-quoted or double-quoted strings are ignored. A string ends at the
      next unescaped quote of the same kind; inside a string, a backslash escapes the next
      character.
    - An unterminated string makes the result False.
    - Outside strings, backslashes have no special meaning.
    Example: balanced('f(a[1], "x)")') is True
    """
''', '''
def balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in "\\'\\"":
            q = c; i += 1
            while i < n and s[i] != q:
                i += 2 if s[i] == "\\\\" else 1
            if i >= n: return False
        elif c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]: return False
        i += 1
    return not stack
''', ["""('f(a[1], "x)")')""", "('([)]')", """("'(' + ']'")""", """('"abc')""", "('')",
      """('("a\\\\"b)")')""", """('("a\\\\\\\\")')""", "('\\\\(')", """("{'\\"'}")""", "('((')",
      """('("\\'")')"""], 4)

task("eval_expr", '''
def eval_expr(s: str) -> int:
    """Evaluate an integer expression with + - * / ^, unary minus, parentheses and spaces.
    - "/" is integer division that truncates toward zero (7 / -2 == -3); division by zero
      raises ZeroDivisionError.
    - "^" is exponentiation (non-negative exponents only), binds tighter than unary minus and
      is right-associative: -2^2 == -4, 2^3^2 == 512.
    - Unary minus binds tighter than * and /: -3*2 == -6, 2*-3 == -6, --3 == 3.
    - * and / are left-associative and bind tighter than + and -, which are left-associative.
    Do not use eval().
    Example: eval_expr("1 + 2 * 3") == 7
    """
''', '''
def eval_expr(s):
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace(): i += 1; continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit(): j += 1
            toks.append(int(s[i:j])); i = j
        else:
            toks.append(c); i += 1
    pos = 0
    def peek():
        return toks[pos] if pos < len(toks) else None
    def take():
        nonlocal pos; pos += 1; return toks[pos - 1]
    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take(); r = term(); v = v + r if op == "+" else v - r
        return v
    def term():
        v = unary()
        while peek() in ("*", "/"):
            op = take(); r = unary()
            if op == "*": v *= r
            else:
                if r == 0: raise ZeroDivisionError
                q = abs(v) // abs(r)
                v = q if (v >= 0) == (r >= 0) else -q
        return v
    def unary():
        if peek() == "-":
            take(); return -unary()
        return power()
    def power():
        b = atom()
        if peek() == "^":
            take(); e = unary_pow(); return b ** e
        return b
    def unary_pow():
        if peek() == "-":
            take(); return -unary_pow()
        return power()
    def atom():
        t = take()
        if t == "(":
            v = expr(); take(); return v
        return t
    return expr()
''', ['("1 + 2 * 3")', '("7 / -2")', '("-2^2")', '("2^3^2")', '("2*-3")', '("--3")', '("(1+2)*(3-4)")',
      '("10 - 4 - 3")', '("-7 / 2")', '("1/0")', '("(-2)^2")', '("2^2*3")', '("100 / 7 / 2")',
      '("-(2+3)^2")'], 5)

task("accounting_format", '''
def accounting_format(cents: int) -> str:
    """Format an amount in cents in accounting style.
    - Dollars use comma thousands separators and always two decimals: 123456 -> "1,234.56".
    - Negative amounts are wrapped in parentheses instead of a minus sign: -5 -> "(0.05)".
    - Zero is "-" (a single hyphen).
    Example: accounting_format(-123456789) == "(1,234,567.89)"
    """
''', '''
def accounting_format(cents):
    if cents == 0: return "-"
    a = abs(cents)
    s = f"{a // 100:,}.{a % 100:02d}"
    return f"({s})" if cents < 0 else s
''', ["(-123456789)", "(123456)", "(0)", "(-5)", "(100)", "(99999)", "(1)", "(-100000)"], 3)

task("dedupe_keep_last", '''
def dedupe_keep_last(items: list) -> list:
    """Remove duplicates, keeping the LAST occurrence of each value, and preserve the relative
    order of the kept elements. Items are hashable.
    Example: dedupe_keep_last([1, 2, 1, 3, 2]) == [1, 3, 2]
    """
''', '''
def dedupe_keep_last(items):
    last = {x: i for i, x in enumerate(items)}
    return [x for i, x in enumerate(items) if last[x] == i]
''', ["([1, 2, 1, 3, 2])", "([])", "(['a', 'b', 'a'])", "([1, 1, 1])", "([3, 1, 2, 3, 1])",
      "([1, True, 2])", "([(1, 2), (1, 2), 0])"], 3)

task("normalize_whitespace", '''
def normalize_whitespace(s: str) -> str:
    """Normalize whitespace (characters for which str.isspace() is True).
    - Leading and trailing whitespace is removed.
    - Every other run of whitespace is replaced by:
        "\\n\\n" if the run contains two or more newlines ("\\n"),
        "\\n" if it contains exactly one newline,
        " " otherwise.
    Example: normalize_whitespace("  a \\t b\\n\\n\\n c ") == "a b\\n\\nc"
    """
''', '''
import re
def normalize_whitespace(s):
    def rep(m):
        n = m.group().count("\\n")
        return "\\n\\n" if n >= 2 else "\\n" if n == 1 else " "
    return re.sub(r"\\s+", rep, s.strip())
''', ['("  a \\t b\\n\\n\\n c ")', '("a\\nb")', '("a \\n \\n b")', '("")', '("a\\r\\nb")',
      '("   ")', '("x\\u00a0y")', '("a\\n\\tb  c")', '("one\\x0btwo")'], 3)

task("count_islands", '''
def count_islands(grid: list[str]) -> int:
    """Count islands in a grid of "1" (land) and "0" (water) given as a list of equal-length
    strings. Land cells are connected horizontally, vertically AND diagonally (8 neighbours).
    Example: count_islands(["110", "001", "000"]) == 1
    """
''', '''
def count_islands(grid):
    seen, n = set(), 0
    R = len(grid); C = len(grid[0]) if grid else 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == "1" and (r, c) not in seen:
                n += 1; stack = [(r, c)]; seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < R and 0 <= xx < C and grid[yy][xx] == "1" and (yy, xx) not in seen:
                                seen.add((yy, xx)); stack.append((yy, xx))
    return n
''', ['(["110", "001", "000"])', '([])', '(["101", "000", "101"])', '(["10", "01"])', '(["0"])',
      '(["1001", "0110", "1001"])', '(["111", "101", "111"])'], 3)

task("longest_palindrome", '''
def longest_palindrome(s: str) -> str:
    """Return the longest palindromic substring of s (case-sensitive).
    If several have the maximum length, return the one that starts first. "" -> "".
    Example: longest_palindrome("babad") == "bab"
    """
''', '''
def longest_palindrome(s):
    best = ""
    for i in range(len(s)):
        for j in range(i + len(best) + 1, len(s) + 1):
            t = s[i:j]
            if t == t[::-1] and len(t) > len(best): best = t
    return best
''', ['("babad")', '("")', '("cbbd")', '("abc")', '("Aa")', '("abacdfgdcaba")', '("aaaa")', '("xabay")'], 3)

task("title_case", '''
def title_case(s: str) -> str:
    """Convert a sentence to title case. Words are separated by single spaces.
    - Capitalize each word (first letter upper, rest lower), except the small words
      a, an, the, and, but, or, for, nor, on, at, to, by, of, in, which are all lowercase
      unless they are the first or last word.
    - Treat each hyphen-separated part of a word as its own word for capitalization
      ("well-known" -> "Well-Known"); the small-word rule does not apply to parts of a
      hyphenated word, and "first or last word" refers to whole words.
    - A word (or hyphen part) that is entirely uppercase letters and at least two characters
      long, like "NASA", is kept as it is.
    Example: title_case("the lord of the rings") == "The Lord of the Rings"
    """
''', '''
SMALL = {"a","an","the","and","but","or","for","nor","on","at","to","by","of","in"}
def title_case(s):
    words = s.split(" ")
    out = []
    for i, w in enumerate(words):
        if "-" in w:
            out.append("-".join(cap(p) for p in w.split("-")))
        elif w.lower() in SMALL and 0 < i < len(words) - 1 and not (len(w) >= 2 and w.isupper() and w.isalpha()):
            out.append(w.lower())
        else:
            out.append(cap(w))
    return " ".join(out)
def cap(p):
    if len(p) >= 2 and p.isalpha() and p.isupper(): return p
    return p[:1].upper() + p[1:].lower()
''', ['("the lord of the rings")', '("a tale of two cities")', '("what is it for")', '("NASA and the moon")',
      '("a well-known end-of-year party")', '("hELLO wORLD")', '("war and peace")', '("x")',
      '("the IT crowd in A city")', '("go to")'], 4)

task("parse_query", '''
def parse_query(qs: str) -> dict[str, list[str]]:
    """Parse a URL query string into {key: [values in order]}.
    - An optional leading "?" is ignored. Pairs are separated by "&"; empty pairs are skipped.
    - Each pair splits at the first "="; a pair without "=" has the value "".
    - In keys and values, "+" means a space and %XX sequences are percent-decoded as UTF-8.
    - Keys keep their first-appearance order in the dict.
    Example: parse_query("a=1&b=2&a=3") == {"a": ["1", "3"], "b": ["2"]}
    """
''', '''
from urllib.parse import unquote_plus
def parse_query(qs):
    if qs.startswith("?"): qs = qs[1:]
    out = {}
    for pair in qs.split("&"):
        if not pair: continue
        k, _, v = pair.partition("=")
        out.setdefault(unquote_plus(k), []).append(unquote_plus(v))
    return out
''', ['("a=1&b=2&a=3")', '("")', '("?x=hello+world")', '("a=1&&b=2")', '("flag&x=")', '("k=a=b")',
      '("name=%C3%A9t%C3%A9")', '("%2B=%26")', '("?")', '("a+b=c%20d")'], 4)

task("chunk_by_sum", '''
def chunk_by_sum(nums: list[int], limit: int) -> list[list[int]]:
    """Split a list of positive integers into consecutive groups, greedily: add each number to
    the current group while the group's sum stays <= limit, otherwise start a new group.
    A number larger than limit forms a group on its own. [] -> [].
    Example: chunk_by_sum([3, 4, 2, 5, 1], 7) == [[3, 4], [2, 5], [1]]
    """
''', '''
def chunk_by_sum(nums, limit):
    out, cur, s = [], [], 0
    for x in nums:
        if cur and s + x > limit:
            out.append(cur); cur, s = [], 0
        cur.append(x); s += x
        if s > limit:
            out.append(cur); cur, s = [], 0
    if cur: out.append(cur)
    return out
''', ["([3, 4, 2, 5, 1], 7)", "([], 5)", "([10, 1, 1], 5)", "([1, 1, 10, 1], 5)", "([5, 5], 5)",
      "([2, 2, 2], 3)", "([6], 5)"], 3)

task("common_dir", '''
def common_dir(paths: list[str]) -> str:
    """Return the longest common prefix of the paths, compared component by component
    ("/a/bc" and "/a/bd" share "/a", not "/a/b").
    - Paths use "/" separators; repeated and trailing slashes are ignored ("/a//b/" == "/a/b").
    - All absolute: the result is absolute ("/" if only the root is shared).
      All relative: the result is relative ("" if nothing is shared).
      A mix of absolute and relative paths, or an empty list, gives "".
    - A single path returns that path (normalized).
    Example: common_dir(["/a/b/c", "/a/b/d"]) == "/a/b"
    """
''', '''
def common_dir(paths):
    if not paths: return ""
    abss = {p.startswith("/") for p in paths}
    if len(abss) > 1: return ""
    parts = [[c for c in p.split("/") if c] for p in paths]
    common = []
    for comps in zip(*parts):
        if all(c == comps[0] for c in comps): common.append(comps[0])
        else: break
    s = "/".join(common)
    return "/" + s if abss.pop() else s
''', ['(["/a/b/c", "/a/b/d"])', '(["/a/bc", "/a/bd"])', '([])', '(["/x", "y"])', '(["/a//b/"])',
      '(["/a", "/b"])', '(["a/b", "a/c"])', '(["a", "b"])', '(["/a/b", "/a/b/c"])'], 4)

task("compress_ranges", '''
def compress_ranges(nums: list[int]) -> str:
    """Describe a set of integers compactly. Duplicates are ignored and order does not matter.
    - Sort the distinct values; write runs of 3 or more consecutive integers as "start-end";
      write every other value on its own. Separate items with ",".
    - Two consecutive numbers are written separately: [7, 8] -> "7,8".
    - Negative numbers keep their sign: [-3, -2, -1] -> "-3--1". [] -> "".
    Example: compress_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7,8"
    """
''', '''
def compress_ranges(nums):
    v = sorted(set(nums)); out = []; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[j] + 1: j += 1
        if j - i >= 2: out.append(f"{v[i]}-{v[j]}")
        else: out.extend(str(x) for x in v[i:j + 1])
        i = j + 1
    return ",".join(out)
''', ["([1, 2, 3, 5, 7, 8])", "([])", "([-3, -2, -1])", "([5, 3, 4, 4])", "([7, 8])", "([1, 3, 5])",
      "([-1, 0, 1, 2, 10, 11])", "([0])"], 3)

task("day_of_year", '''
def day_of_year(s: str) -> int:
    """Return the day of the year (1-366) of a Gregorian date "YYYY-MM-DD" (exactly 4, 2 and 2
    digits). Leap years: divisible by 4, except centuries, unless divisible by 400.
    Invalid input (bad format, month 0 or 13, day out of range for the month) raises ValueError.
    Do not use the datetime or calendar modules.
    Example: day_of_year("2024-03-01") == 61
    """
''', '''
import re
def day_of_year(s):
    m = re.fullmatch(r"(\\d{4})-(\\d{2})-(\\d{2})", s)
    if not m: raise ValueError(s)
    y, mo, d = map(int, m.groups())
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if not 1 <= mo <= 12 or not 1 <= d <= days[mo - 1]: raise ValueError(s)
    return sum(days[:mo - 1]) + d
''', ['("2024-03-01")', '("2023-03-01")', '("1900-03-01")', '("2000-03-01")', '("2023-02-29")',
      '("2023-13-01")', '("2023-1-01")', '("2023-12-31")', '("2024-12-31")', '("2023-04-31")', '("2023-00-10")'], 4)

task("determinant", '''
def determinant(m: list[list[int]]) -> int:
    """Return the exact determinant of a square integer matrix (up to 10 x 10, entries can be
    very large). The result must be an exact int, so floating point is not acceptable.
    The determinant of the empty matrix [] is 1. A non-square matrix raises ValueError.
    Example: determinant([[1, 2], [3, 4]]) == -2
    """
''', '''
from fractions import Fraction
def determinant(m):
    n = len(m)
    if any(len(r) != n for r in m): raise ValueError
    a = [[Fraction(x) for x in r] for r in m]
    det = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None: return 0
        if p != c: a[c], a[p] = a[p], a[c]; det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n): a[r][k] -= f * a[c][k]
    return int(det)
''', ["([[1, 2], [3, 4]])", "([])", "([[5]])", "([[1, 2, 3], [4, 5, 6]])", "([[0, 1], [1, 0]])",
      "([[10**20, 1], [1, 10**20]])", "([[2, 0, 0], [0, 3, 0], [0, 0, 4]])",
      "([[1, 2, 3], [4, 5, 6], [7, 8, 9]])", "([[i * 7 + j * j + (i == j) * 10**15 for j in range(8)] for i in range(8)])"], 4)

task("bowling_score", '''
def bowling_score(rolls: list[int]) -> int:
    """Score a complete, valid game of ten-pin bowling from the list of pins knocked down per
    roll. A strike scores 10 plus the next two rolls; a spare scores 10 plus the next roll;
    the 10th frame may contain one or two bonus rolls, which count only as bonuses.
    Example: bowling_score([10] * 12) == 300
    """
''', '''
def bowling_score(rolls):
    score, i = 0, 0
    for _ in range(10):
        if rolls[i] == 10:
            score += 10 + rolls[i + 1] + rolls[i + 2]; i += 1
        elif rolls[i] + rolls[i + 1] == 10:
            score += 10 + rolls[i + 2]; i += 2
        else:
            score += rolls[i] + rolls[i + 1]; i += 2
    return score
''', ["([10] * 12)", "([0] * 20)", "([5, 5] * 10 + [5])", "([9, 0] * 10)", "([10, 3, 4] + [0] * 16)",
      "([0] * 18 + [10, 10, 10])", "([0] * 18 + [7, 3, 10])", "([10, 10, 10, 0, 0] + [0] * 12)"], 3)

task("time_ago", '''
def time_ago(seconds: int) -> str:
    """Describe an elapsed time (seconds >= 0) in words, using these rules in order:
    - < 45 s: "just now"
    - < 90 s: "a minute ago"
    - < 45 minutes: "N minutes ago", N = seconds / 60 rounded half up
    - < 90 minutes: "an hour ago"
    - < 22 hours: "N hours ago", N = seconds / 3600 rounded half up
    - < 36 hours: "a day ago"
    - otherwise: "N days ago", N = seconds / 86400 rounded half up
    Example: time_ago(150) == "3 minutes ago"
    """
''', '''
def time_ago(seconds):
    def r(x, u): return (2 * x + u) // (2 * u)
    if seconds < 45: return "just now"
    if seconds < 90: return "a minute ago"
    if seconds < 45 * 60: return f"{r(seconds, 60)} minutes ago"
    if seconds < 90 * 60: return "an hour ago"
    if seconds < 22 * 3600: return f"{r(seconds, 3600)} hours ago"
    if seconds < 36 * 3600: return "a day ago"
    return f"{r(seconds, 86400)} days ago"
''', ["(150)", "(44)", "(45)", "(89)", "(90)", "(2669)", "(2700)", "(5399)", "(5400)", "(9000)",
      "(79199)", "(79200)", "(129600)", "(216000)", "(0)"], 4)

task("caesar_words", '''
def caesar_words(s: str, k: int) -> str:
    """Shift each ASCII letter k places along the alphabet (wrapping around, k may be negative or
    larger than 26), preserving case, EXCEPT that letters in a word starting with "#" are left
    unchanged (the whole word, including the "#"). A word is a maximal run of non-space
    characters. Every other character is left unchanged.
    Example: caesar_words("Hello #World zz", 1) == "Ifmmp #World aa"
    """
''', '''
def caesar_words(s, k):
    def shift(w):
        if w.startswith("#"): return w
        out = []
        for c in w:
            if "a" <= c <= "z": out.append(chr((ord(c) - 97 + k) % 26 + 97))
            elif "A" <= c <= "Z": out.append(chr((ord(c) - 65 + k) % 26 + 65))
            else: out.append(c)
        return "".join(out)
    return " ".join(shift(w) for w in s.split(" "))
''', ['("Hello #World zz", 1)', '("abc", -1)', '("xyz", 29)', '("a#b c", 1)', '("  ab  ", 1)',
      '("Café", 1)', '("#tag", 5)', '("", 3)', '("Zz9", -27)'], 3)

task("matrix_rotate", '''
def matrix_rotate(m: list[list], turns: int) -> list[list]:
    """Rotate a rectangular matrix 90 degrees clockwise `turns` times (turns may be negative,
    meaning counter-clockwise, or larger than 4). Return a new matrix; do not modify the input.
    An empty matrix [] stays []. Rows of unequal length raise ValueError.
    Example: matrix_rotate([[1, 2, 3], [4, 5, 6]], 1) == [[4, 1], [5, 2], [6, 3]]
    """
''', '''
def matrix_rotate(m, turns):
    if any(len(r) != len(m[0]) for r in m): raise ValueError
    a = [list(r) for r in m]
    for _ in range(turns % 4):
        if not a: break
        a = [list(r) for r in zip(*a[::-1])]
    return a
''', ["([[1, 2, 3], [4, 5, 6]], 1)", "([], 3)", "([[1, 2], [3, 4]], -1)", "([[1, 2, 3], [4, 5, 6]], 2)",
      "([[1], [2, 3]], 1)", "([[1, 2, 3]], 7)", "([[1, 2], [3, 4]], 0)", "([[1, 2], [3, 4], [5, 6]], -3)"], 3)

task("anagram_groups", '''
def anagram_groups(words: list[str]) -> list[list[str]]:
    """Group words that are anagrams of each other, ignoring case. Within a group keep the
    words' original spelling and input order; order groups by the position of their first word
    in the input. Each occurrence is kept, including duplicates.
    Example: anagram_groups(["Tea", "eat", "tan", "ate", "nat", "bat"]) == [["Tea", "eat", "ate"], ["tan", "nat"], ["bat"]]
    """
''', '''
def anagram_groups(words):
    groups = {}
    for w in words:
        groups.setdefault("".join(sorted(w.lower())), []).append(w)
    return list(groups.values())
''', ['(["Tea", "eat", "tan", "ate", "nat", "bat"])', '([])', '(["a", "A", "a"])', '(["ab", "ba", "abc", "cab", "b"])',
      '(["", ""])', '(["Listen", "Silent", "enlist", "google"])'], 3)

task("word_freq", '''
def word_freq(text: str, n: int) -> list[tuple[str, int]]:
    """Return the n most frequent words as (word, count) pairs.
    - Words are maximal runs of ASCII letters and apostrophes, lowercased; apostrophes at the
      start or end of a word are stripped, and a word that becomes empty is ignored.
    - Sort by count descending, then alphabetically. If fewer than n words exist, return all.
    Example: word_freq("The cat and the hat.", 2) == [("the", 2), ("and", 1)]
    """
''', '''
import re
from collections import Counter
def word_freq(text, n):
    words = [w.strip("'") for w in re.findall(r"[A-Za-z']+", text.lower())]
    c = Counter(w for w in words if w)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
''', ['("The cat and the hat.", 2)', '("", 3)', '("don\'t stop, don\'t", 5)', '("\'quoted\' words \'", 5)',
      '("b a c b a", 2)', '("rock-and-roll and ROCK", 2)', '("it\'s its it\'s", 1)', '("café latte", 5)'], 3)

assert len(TASKS) == 40, len(TASKS)
