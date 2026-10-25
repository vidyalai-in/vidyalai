# 15 harder, rule-heavy tasks. Executed inside tasks.py (uses its task() helper).

task("cron_matches", '''
def cron_matches(expr: str, when: str) -> bool:
    """Return True if the time `when` ("YYYY-MM-DD HH:MM") matches the cron expression `expr`.
    - expr has five space-separated fields: minute (0-59), hour (0-23), day of month (1-31),
      month (1-12) and day of week (0-7, where 0 and 7 both mean Sunday, 1 is Monday ... 6 is
      Saturday).
    - Each field is a comma-separated list of items. An item is "*", a number "a", a range
      "a-b", or "*" or "a-b" followed by "/n" (every n-th value from the start of the range;
      for "*" the range is the whole field).
    - Day matching: if both the day-of-month and the day-of-week fields are restricted (do not
      start with "*"), the day matches when EITHER of them matches. Otherwise both must match.
    Example: cron_matches("*/15 9-17 * * 1-5", "2026-10-23 09:45") is True
    """
''', '''
from datetime import date
def cron_matches(expr, when):
    f = expr.split()
    d, t = when.split()
    y, mo, da = map(int, d.split("-")); h, mi = map(int, t.split(":"))
    dow = (date(y, mo, da).weekday() + 1) % 7
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    def values(field, lo, hi):
        out = set()
        for item in field.split(","):
            step = 1
            if "/" in item:
                item, s = item.split("/"); step = int(s)
            if item == "*": a, b = lo, hi
            elif "-" in item: a, b = map(int, item.split("-"))
            else: a = b = int(item)
            out.update(range(a, b + 1, step))
        return out
    vals = [values(x, *b) for x, b in zip(f, bounds)]
    vals[4] = {0 if v == 7 else v for v in vals[4]}
    dom_ok, dow_ok = da in vals[2], dow in vals[4]
    if not f[2].startswith("*") and not f[4].startswith("*"):
        day = dom_ok or dow_ok
    else:
        day = dom_ok and dow_ok
    return mi in vals[0] and h in vals[1] and mo in vals[3] and day
''', ['("*/15 9-17 * * 1-5", "2026-10-23 09:45")', '("0 0 * * 0", "2026-10-25 00:00")', '("30 8 1,15 * *", "2026-10-15 08:30")',
      '("0-30/10 * * * *", "2026-10-24 12:35")', '("0 0 13 * 5", "2026-10-23 00:00")', '("0 0 13 * 5", "2026-10-22 00:00")',
      '("0 0 * * 7", "2026-10-25 00:00")', '("0 0 */2 * 5", "2026-10-24 00:00")', '("0 0 */2 * 5", "2026-10-23 00:00")',
      '("5 4 * * 5-7", "2026-10-25 04:05")', '("0 0 1 * 1", "2026-10-26 00:00")', '("0-30/10 * * * *", "2026-10-24 12:20")',
      '("0 12 * 2 *", "2026-02-28 12:00")', '("*/7 * * * *", "2026-10-24 12:56")'], 4)

task("damerau_levenshtein", '''
def damerau_levenshtein(a: str, b: str) -> int:
    """Return the Damerau-Levenshtein distance between a and b: the minimum number of
    single-character insertions, deletions, substitutions and transpositions of two adjacent
    characters needed to turn a into b. This is the unrestricted distance: a substring may be
    edited more than once (for example, characters may be inserted between two characters
    after they have been transposed).
    Example: damerau_levenshtein("kitten", "sitting") == 3
    """
''', '''
def damerau_levenshtein(a, b):
    INF = len(a) + len(b)
    da = {}
    d = [[0] * (len(b) + 2) for _ in range(len(a) + 2)]
    d[0][0] = INF
    for i in range(len(a) + 1): d[i + 1][0] = INF; d[i + 1][1] = i
    for j in range(len(b) + 1): d[0][j + 1] = INF; d[1][j + 1] = j
    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            i1 = da.get(b[j - 1], 0); j1 = db
            cost = 1
            if a[i - 1] == b[j - 1]: cost = 0; db = j
            d[i + 1][j + 1] = min(d[i][j] + cost, d[i + 1][j] + 1, d[i][j + 1] + 1,
                                  d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1))
        da[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]
''', ['("kitten", "sitting")', '("ab", "ba")', '("", "abc")', '("teh", "the")', '("CA", "ABC")', '("abc", "ca")',
      '("a cat", "an act")', '("abcdef", "badcfe")', '("ab", "bca")', '("abcd", "dacb")', '("same", "same")'], 4)

task("expand_braces", '''
def expand_braces(s: str) -> list[str]:
    """Expand braces like a Unix shell, returning the words in order.
    - "{x,y,...}" with at least one comma at its own nesting level expands to each
      alternative (alternatives may be empty or contain further braces):
      "a{b,c}d" -> ["abd", "acd"], "a{,b}" -> ["a", "ab"].
    - "{a..b}" and "{a..b..n}" are sequences from a to b inclusive in steps of |n| (n = 0 means
      1), counting down if a > b. a and b are both integers or both single letters. If either
      integer endpoint is written with a leading zero (like 01), every number is zero-padded to
      the width of the longer endpoint as written: "{01..10..3}" -> ["01", "04", "07", "10"].
    - Several brace groups multiply; the leftmost group varies slowest.
    - A "{" with no matching "}", or a brace pair that is neither a valid list nor a valid
      sequence (like "{}", "{a}" or "{1..b}"), is literal text; braces inside it can still
      expand: "{x{a,b}}" -> ["{xa}", "{xb}"].
    Example: expand_braces("x{a,b}{1..2}") == ["xa1", "xa2", "xb1", "xb2"]
    """
''', '''
import re
def _seq(body):
    m = re.fullmatch(r"(-?\\d+)\\.\\.(-?\\d+)(?:\\.\\.(-?\\d+))?", body)
    if m:
        a, b = int(m[1]), int(m[2]); step = abs(int(m[3])) if m[3] else 1; step = step or 1
        width = max(len(m[1]), len(m[2])) if any(re.fullmatch(r"-?0\\d+", x) for x in (m[1], m[2])) else 0
        rng = range(a, b + 1, step) if a <= b else range(a, b - 1, -step)
        return [str(v).zfill(width) for v in rng]
    m = re.fullmatch(r"([A-Za-z])\\.\\.([A-Za-z])(?:\\.\\.(-?\\d+))?", body)
    if m:
        a, b = ord(m[1]), ord(m[2]); step = abs(int(m[3])) if m[3] else 1; step = step or 1
        rng = range(a, b + 1, step) if a <= b else range(a, b - 1, -step)
        return [chr(v) for v in rng]
    return None
def expand_braces(s):
    i = 0
    while i < len(s):
        if s[i] == "{":
            depth, j = 0, i
            while j < len(s):
                if s[j] == "{": depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0: break
                j += 1
            if j < len(s):
                body = s[i + 1:j]
                parts, depth, start = [], 0, 0
                for k, c in enumerate(body):
                    if c == "{": depth += 1
                    elif c == "}": depth -= 1
                    elif c == "," and depth == 0:
                        parts.append(body[start:k]); start = k + 1
                parts.append(body[start:])
                pre, post = s[:i], s[j + 1:]
                alts = parts if len(parts) >= 2 else _seq(body)
                if alts is not None:
                    return [x for p in alts for x in expand_braces(pre + p + post)]
        i += 1
    return [s]
''', ['("x{a,b}{1..2}")', '("a{b,c}d")', '("{1..3}")', '("{a}")', '("{5..1..2}")', '("{01..10..3}")', '("a{,b}")',
      '("{a,b{c,d}}")', '("{x{a,b}}")', '("{}")', '("{c..a}")', '("{1..b}")', '("}{a,b}{")', '("")', '("{-1..2}")'], 5)

task("semver_satisfies", '''
def semver_satisfies(version: str, rng: str) -> bool:
    """Check whether version ("MAJOR.MINOR.PATCH", non-negative integers) satisfies rng.
    - rng is one or more comparator sets separated by "||" (spaces around it are allowed);
      the version must satisfy ANY set. A set is one or more comparators separated by spaces,
      all of which must hold.
    - Comparators: ">=V", "<=V", ">V", "<V", "=V", "V" (same as "=V"), "~V", "^V" and "*"
      (matches anything). V is always a full MAJOR.MINOR.PATCH version, written directly after
      the operator.
    - "~V" allows patch-level changes: ~1.2.3 means >=1.2.3 <1.3.0.
    - "^V" allows changes that do not modify the left-most non-zero component of V:
      ^1.2.3 means >=1.2.3 <2.0.0.
    - Versions compare numerically, component by component.
    Example: semver_satisfies("1.4.0", "^1.2.3") is True
    """
''', '''
def semver_satisfies(version, rng):
    v = tuple(map(int, version.split(".")))
    def ok(comp):
        if comp == "*": return True
        for op in (">=", "<=", ">", "<", "=", "^", "~"):
            if comp.startswith(op):
                w = tuple(map(int, comp[len(op):].split("."))); break
        else:
            op, w = "=", tuple(map(int, comp.split(".")))
        if op == ">=": return v >= w
        if op == "<=": return v <= w
        if op == ">": return v > w
        if op == "<": return v < w
        if op == "=": return v == w
        if op == "~": return w <= v < (w[0], w[1] + 1, 0)
        if w[0]: hi = (w[0] + 1, 0, 0)
        elif w[1]: hi = (0, w[1] + 1, 0)
        else: hi = (0, 0, w[2] + 1)
        return w <= v < hi
    return any(all(ok(c) for c in s.split()) for s in rng.split("||"))
''', ['("1.4.0", "^1.2.3")', '("2.0.0", "^1.2.3")', '("1.10.0", ">1.9.0")', '("1.2.9", "~1.2.3")',
      '("0.2.5", "^0.2.3")', '("0.3.0", "^0.2.3")', '("0.0.4", "^0.0.3")', '("0.0.3", "^0.0.3")', '("1.3.0", "~1.2.3")',
      '("1.5.0", ">=1.0.0 <1.5.0 || =2.0.0")', '("2.0.0", ">=1.0.0 <1.5.0||=2.0.0")', '("3.0.0", "*")', '("1.2.3", "1.2.3")',
      '("1.2.4", "<1.2.4 || >1.2.4")', '("0.1.0", "^0.0.3")', '("0.2.10", "~0.2.9")'], 4)

task("poker_compare", '''
def poker_compare(hand1: str, hand2: str) -> int:
    """Compare two five-card poker hands; return 1 if hand1 wins, -1 if hand2 wins, 0 for a tie.
    - Cards are written rank then suit, separated by spaces: "AS KD 3H 3C 9S". Ranks are
      2-9, T, J, Q, K, A; suits are C, D, H, S and never break ties.
    - Categories from best to worst: straight flush, four of a kind, full house, flush,
      straight, three of a kind, two pair, one pair, high card.
    - Within a category, compare the ranks that matter in order of importance: grouped cards
      first (larger groups first, then higher rank), then the remaining cards from high to low.
    - A-2-3-4-5 is a straight (and straight flush) whose highest card is the 5; A also makes
      the top straight T-J-Q-K-A. Straights compare by their highest card.
    Example: poker_compare("2H 2D 5S 9C KD", "2C 3H 4S 8C AH") == 1
    """
''', '''
def poker_compare(hand1, hand2):
    def score(h):
        cards = h.split()
        ranks = sorted(("23456789TJQKA".index(c[0]) + 2 for c in cards), reverse=True)
        flush = len({c[1] for c in cards}) == 1
        counts = {r: ranks.count(r) for r in ranks}
        groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
        shape = [c for _, c in groups]; order = [r for r, _ in groups]
        uniq = sorted(set(ranks), reverse=True)
        straight = len(uniq) == 5 and uniq[0] - uniq[4] == 4
        if uniq == [14, 5, 4, 3, 2]: straight, order = True, [5, 4, 3, 2, 1]
        if straight and flush: cat = 8
        elif shape == [4, 1]: cat = 7
        elif shape == [3, 2]: cat = 6
        elif flush: cat = 5
        elif straight: cat = 4
        elif shape == [3, 1, 1]: cat = 3
        elif shape == [2, 2, 1]: cat = 2
        elif shape == [2, 1, 1, 1]: cat = 1
        else: cat = 0
        return (cat, order)
    a, b = score(hand1), score(hand2)
    return (a > b) - (a < b)
''', ['("2H 2D 5S 9C KD", "2C 3H 4S 8C AH")', '("AH 2D 3C 4S 5H", "2H 3D 4C 5S 6H")', '("AH 2D 3C 4S 5H", "KH KD KC 2S 3H")',
      '("2H 2D 2S KC KD", "3H 3D 3S 4C 4D")', '("AH KH 9H 4H 2H", "AS KS 8S 7S 6S")', '("TS JS QS KS AS", "9H TH JH QH KH")',
      '("AS 2S 3S 4S 5S", "KD QD JD TD 9D")', '("TH JD QC KS AH", "AD 2C 3H 4D 5C")', '("AH AD 4C 3S 2H", "KH KD QC JS 9H")',
      '("8H 8D 3C 3S KH", "8S 8C 3D 3H QH")', '("5H 5D 5C 9S 9H", "5S 5C 5D 9C 9D")', '("2H 3H 4H 5H 7D", "2C 3C 4C 5C 7S")',
      '("KH KD 7C 7S AH", "KS KC 7D 7H QS")'], 4)

task("next_palindrome", '''
def next_palindrome(n: int) -> int:
    """Return the smallest palindromic integer strictly greater than n (n >= 0). A palindrome
    reads the same forwards and backwards in base 10. n can have up to 50 digits, so the
    function must be fast for very large n.
    Example: next_palindrome(12921) == 13031
    """
''', '''
def next_palindrome(n):
    m = n + 1; s = str(m); L = len(s); k = (L + 1) // 2
    half = s[:k]
    cand = int(half + half[:L // 2][::-1])
    if cand < m:
        half = str(int(half) + 1)
        cand = int(half + half[:L // 2][::-1])
    return cand
''', ["(12921)", "(9)", "(0)", "(99)", "(123)", "(808)", "(999)", "(10**30)", "(191)", "(10**40 - 1)",
      "(1234567890987654321)", "(12345678909876543210)", "(4 * 10**49 + 5)"], 4)

task("json_pointer", '''
def json_pointer(doc, pointer: str):
    """Resolve a JSON Pointer (RFC 6901) against doc (nested dicts and lists).
    - "" refers to the whole document. Any other pointer must start with "/", otherwise raise
      ValueError. The text after each "/" is a reference token; tokens use RFC 6901 escaping
      (~0 encodes "~" and ~1 encodes "/"). A "~" followed by anything else raises ValueError.
    - For a dict, the token is the key. For a list, the token must be "0" or a decimal number
      without leading zeros that is a valid index.
    - If anything referenced does not exist (missing key, bad or out-of-range index, "-",
      or indexing into a non-container), raise LookupError (KeyError and IndexError count).
    Example: json_pointer({"a": [10, 20]}, "/a/1") == 20
    """
''', '''
def json_pointer(doc, pointer):
    if pointer == "": return doc
    if not pointer.startswith("/"): raise ValueError(pointer)
    cur = doc
    for tok in pointer[1:].split("/"):
        i = 0
        while True:
            i = tok.find("~", i)
            if i == -1: break
            if i + 1 >= len(tok) or tok[i + 1] not in "01": raise ValueError(pointer)
            i += 2
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if tok not in cur: raise LookupError(tok)
            cur = cur[tok]
        elif isinstance(cur, list):
            if not (tok == "0" or (tok.isdigit() and tok[0] != "0" and tok.isascii())): raise LookupError(tok)
            if int(tok) >= len(cur): raise LookupError(tok)
            cur = cur[int(tok)]
        else:
            raise LookupError(tok)
    return cur
''', [f'({d}, {p!r})' for d, p in [(D, x) for D in ['{"foo": ["bar", "baz"], "": 0, "a/b": 1, "m~n": 8, " ": 7, "~1": 9}']
      for x in ["/foo/1", "", "/a~1b", "foo", "/m~0n", "/~01", "/", "/foo/01", "/foo/-", "/foo/2", "/ ", "/foo/0/x", "/x~2", "/foo/0", "/~"]]], 4)

task("parse_ini", '''
def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse INI text into {section: {key: value}}. Process lines (split on "\\n") in order and
    classify each line with the first rule that applies:
    1. Blank (only whitespace): ignored.
    2. Continuation: starts with a space or tab AND the previous line was a key line or a
       continuation line. Its stripped text is appended to the previous key's value after "\\n".
    3. Comment: first non-whitespace character is ";" or "#". Ignored.
    4. Section header: after stripping, starts with "[" and ends with "]". The name is the text
       between the brackets, stripped. Names are case-sensitive; a repeated header continues
       the same section. A section appears in the result even if it has no keys.
    5. Key line: contains "=". Split at the first "="; strip key and value; lowercase the key.
       A repeated key overwrites the earlier value. There are no inline comments.
    6. Anything else raises ValueError.
    Keys before the first header go into the section "" (present only if such a key exists).
    Example: parse_ini("[a]\\nX = 1") == {"a": {"x": "1"}}
    """
''', '''
def parse_ini(text):
    out, cur, prev_key = {}, None, None
    for line in text.split("\\n"):
        if not line.strip():
            prev_key = None; continue
        if line[0] in " \\t" and prev_key is not None:
            out[cur][prev_key] += "\\n" + line.strip(); continue
        s = line.strip()
        if s[0] in ";#":
            prev_key = None; continue
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1].strip(); out.setdefault(cur, {}); prev_key = None; continue
        if "=" in line:
            k, v = line.split("=", 1)
            if cur is None:
                cur = ""; out.setdefault("", {})
            k = k.strip().lower(); out[cur][k] = v.strip(); prev_key = k; continue
        raise ValueError(line)
    return out
''', ['("[a]\\nX = 1")', '("top=1\\n[s]\\nk = v ; not a comment")', '("[s]\\nk = a\\n  b\\n\\tc")', '("[s]\\njunk")',
      '("[s]\\nk=1\\n\\n  ; indented comment")', '("[s]\\nk=1\\n  ; part of value")', '("[S]\\n[s]\\nA=1\\na=2")',
      '("[ x ]\\nk = =v= ")', '("[a]\\nk=1\\n[b]\\n[a]\\nj=2")', '("# c\\n[a]\\n# k=1\\n")', '("  [a]  \\nk=1")',
      '("[a]\\nk=1\\n# c\\n  x=2")', '("")'], 4)

task("regex_match", '''
def regex_match(pattern: str, text: str) -> bool:
    """Return True if the whole text matches pattern, in this small regex language:
    - An atom is a literal character, "." (any one character), or "\\\\" followed by any
      character (that character literally). A "\\\\" at the very end is a literal backslash.
    - "*", "+" and "?" after an atom mean zero-or-more, one-or-more and zero-or-one of that
      atom. A "*", "+" or "?" that does not directly follow an atom (at the start of the
      pattern, or right after another quantifier) is a literal character atom itself.
    - Everything else is literal. There are no groups, classes or anchors.
    Example: regex_match("a*b.", "aaabc") is True
    """
''', '''
from functools import lru_cache
def regex_match(pattern, text):
    toks, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\\\":
            toks.append([("lit", pattern[i + 1] if i + 1 < len(pattern) else "\\\\"), None]); i += 2; continue
        if c in "*+?" and toks and toks[-1][1] is None:
            toks[-1][1] = c; i += 1; continue
        toks.append([("any", None) if c == "." else ("lit", c), None]); i += 1
    flat = []
    for atom, q in toks:
        if q == "+": flat += [(atom, None), (atom, "*")]
        else: flat.append((atom, q))
    @lru_cache(None)
    def m(ti, si):
        if ti == len(flat): return si == len(text)
        (kind, ch), q = flat[ti]
        one = si < len(text) and (kind == "any" or text[si] == ch)
        if q is None: return one and m(ti + 1, si + 1)
        if q == "?": return m(ti + 1, si) or (one and m(ti + 1, si + 1))
        return m(ti + 1, si) or (one and m(ti, si + 1))
    return m(0, 0)
''', ['("a*b.", "aaabc")', '("a.c", "abc")', '("\\\\.", ".")', '("a+", "")', '("*a", "*a")', '("a**", "aa*")', '("a?b?c", "c")',
      '(".*", "")', '("a+?", "aa?")', '("x\\\\", "x\\\\")', '("ab", "abc")', '("\\\\.", "a")', '("a.*b.*c", "axxbyyc")',
      '("?", "?")', '("\\\\*+", "***")', '("a+*", "a*")'], 4)

task("spreadsheet_eval", '''
def spreadsheet_eval(cells: dict[str, str]) -> dict[str, int | str]:
    """Evaluate a tiny spreadsheet. Keys are cell names (a capital letter then digits, "A1").
    Each value is an integer literal (possibly negative, "-5") or a formula: "=" followed by
    integers, cell names, + - * / and parentheses (no unary minus inside formulas).
    - Usual precedence (* and / before + and -), left-associative. "/" is integer division
      truncating toward zero.
    - A reference to a cell that is not in the dict counts as 0.
    - A cell that is part of a reference cycle, or references (directly or indirectly) a cell
      in a cycle, evaluates to "#CYCLE!".
    - Otherwise, a cell that divides by zero, or references (directly or indirectly) a cell
      that does, evaluates to "#DIV/0!".
    Return a dict with the value of every input cell (ints or the error strings).
    Example: spreadsheet_eval({"A1": "5", "B1": "=A1*2"}) == {"A1": 5, "B1": 10}
    """
''', '''
import re
def spreadsheet_eval(cells):
    refs = {k: set(re.findall(r"[A-Z]\\d+", v)) if v.startswith("=") else set() for k, v in cells.items()}
    state, bad = {}, set()
    def dfs(u, stack):
        state[u] = 1; stack.append(u)
        for w in refs[u]:
            if w not in cells: continue
            if state.get(w) == 1:
                bad.update(stack[stack.index(w):])
            elif state.get(w) is None:
                dfs(w, stack)
        stack.pop(); state[u] = 2
    for k in cells:
        if k not in state: dfs(k, [])
    changed = True
    while changed:
        changed = False
        for k in cells:
            if k not in bad and any(w in bad for w in refs[k]):
                bad.add(k); changed = True
    memo = {}
    def val(k):
        if k not in cells: return 0
        if k in bad: return "#CYCLE!"
        if k in memo: return memo[k]
        v = cells[k]
        if not v.startswith("="):
            memo[k] = int(v); return memo[k]
        toks = re.findall(r"\\d+|[A-Z]\\d+|[-+*/()]", v[1:])
        pos = [0]
        class Div(Exception): pass
        def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
        def take(): pos[0] += 1; return toks[pos[0] - 1]
        def expr():
            x = term()
            while peek() in ("+", "-"):
                op = take(); y = term(); x = x + y if op == "+" else x - y
            return x
        def term():
            x = atom()
            while peek() in ("*", "/"):
                op = take(); y = atom()
                if op == "*": x *= y
                else:
                    if y == 0: raise Div
                    q = abs(x) // abs(y); x = q if (x >= 0) == (y >= 0) else -q
            return x
        def atom():
            t = take()
            if t == "(":
                x = expr(); take(); return x
            if t[0].isdigit(): return int(t)
            r = val(t)
            if isinstance(r, str): raise Div
            return r
        try:
            memo[k] = expr()
        except Div:
            memo[k] = "#DIV/0!"
        return memo[k]
    return {k: val(k) for k in cells}
''', ['({"A1": "5", "B1": "=A1*2"})', '({"A1": "5", "B1": "=A1*2", "C1": "=B1+A1*(2-3)"})',
      '({"A1": "=B1", "B1": "=A1", "C1": "=A1+1", "D1": "7"})', '({"A1": "=1/0", "B1": "=A1+1", "C1": "=7/2"})',
      '({"A1": "=(0-7)/2", "B1": "=Z9+1", "C1": "-5"})', '({"A1": "=A1"})', '({"A1": "=1/0", "B1": "=B1", "C1": "=A1+B1"})',
      '({"A1": "=B1*C1", "B1": "=C1-1", "C1": "3", "D1": "=A1/B1"})', '({"A1": "=10-4-3", "B1": "=100/7/2", "C1": "=2*3+4*5"})',
      '({"A1": "=B1", "B1": "=C1", "C1": "=D1", "D1": "=B1", "E1": "=A1/0"})', '({"A2": "=A10/A3", "A3": "=A10*0"})'], 4)

task("rate_limiter", '''
def rate_limiter(timestamps: list[int], limit: int, window: int) -> list[bool]:
    """Decide which requests a sliding-window rate limiter allows.
    Requests arrive at the given integer timestamps (seconds, non-decreasing). A request at time
    t is allowed if fewer than `limit` previously ALLOWED requests have timestamps in the
    half-open interval (t - window, t]. Rejected requests do not count toward the limit.
    Return one boolean per request.
    Example: rate_limiter([0, 1, 2], 2, 10) == [True, True, False]
    """
''', '''
from collections import deque
def rate_limiter(timestamps, limit, window):
    q, out = deque(), []
    for t in timestamps:
        while q and q[0] <= t - window: q.popleft()
        if len(q) < limit:
            q.append(t); out.append(True)
        else:
            out.append(False)
    return out
''', ["([0, 1, 2], 2, 10)", "([], 3, 5)", "([0, 0, 0, 1, 10, 10, 11], 2, 10)", "([0, 5, 10, 15], 1, 10)",
      "([1, 2, 3, 11, 12, 13], 2, 10)", "([0, 9, 10, 19, 20], 2, 10)", "([5, 5, 5], 0, 10)", "([0, 1, 1, 2, 3], 3, 1)"], 3)

task("natural_sort", '''
def natural_sort(items: list[str]) -> list[str]:
    """Sort strings in natural order and return a new list.
    - Split each string into maximal runs of ASCII digits and runs of other characters.
    - Compare two strings run by run: two digit runs compare by numeric value; two text runs
      compare case-insensitively (str.lower()); a digit run sorts before a text run. If all
      compared runs are equal, the string with fewer runs sorts first.
    - Strings that are still equal (like "a01" and "a1", or "A" and "a") are ordered by plain
      Python string comparison.
    Example: natural_sort(["file10", "file9", "File1"]) == ["File1", "file9", "file10"]
    """
''', '''
import re
def natural_sort(items):
    def key(s):
        return ([(0, int(r), "") if r.isdigit() else (1, 0, r.lower()) for r in re.findall(r"[0-9]+|[^0-9]+", s)], s)
    return sorted(items, key=key)
''', ['(["file10", "file9", "File1"])', '([])', '(["b", "A", "a"])', '(["x", "x1", "1"])', '(["a01", "a1", "a001"])',
      '(["v1.10", "v1.9", "v1.9a"])', '(["", "a"])', '(["2", "10", "1a", "1"])', '(["a2b", "a2", "a10b1"])',
      '(["B10", "b2", "a100"])', '(["x0", "x00", "x"])'], 4)

task("topo_order", '''
def topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """Return the lexicographically smallest topological order of nodes 0..n-1, where an edge
    (u, v) means u must come before v. Edges may repeat. If the graph has a cycle (including a
    self-loop), return None. n = 0 gives [].
    Example: topo_order(3, [(2, 0)]) == [1, 2, 0]
    """
''', '''
import heapq
def topo_order(n, edges):
    adj = [set() for _ in range(n)]; indeg = [0] * n
    for u, v in set(edges):
        if v not in adj[u]:
            adj[u].add(v); indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(h); out = []
    while h:
        u = heapq.heappop(h); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(h, v)
    return out if len(out) == n else None
''', ["(3, [(2, 0)])", "(0, [])", "(2, [(0, 1), (1, 0)])", "(3, [(1, 1)])", "(4, [(3, 0), (3, 0), (2, 1)])",
      "(5, [(4, 1), (1, 0), (3, 0)])", "(3, [])", "(6, [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)])"], 3)

task("free_slots", '''
def free_slots(busy_a: list[str], busy_b: list[str], day_start: str, day_end: str, minutes: int) -> list[str]:
    """Find meeting slots when both people are free.
    - Times are "HH:MM" (24-hour). Busy periods are "HH:MM-HH:MM" half-open intervals; they may
      overlap, be unsorted, or extend outside the day.
    - Return every maximal interval inside [day_start, day_end) during which neither person is
      busy and whose length is at least `minutes`, formatted "HH:MM-HH:MM", in time order.
    Example: free_slots(["09:00-10:00"], ["10:30-11:00"], "09:00", "12:00", 30) == ["10:00-10:30", "11:00-12:00"]
    """
''', '''
def free_slots(busy_a, busy_b, day_start, day_end, minutes):
    def t(s): h, m = map(int, s.split(":")); return h * 60 + m
    def f(x): return f"{x // 60:02d}:{x % 60:02d}"
    lo, hi = t(day_start), t(day_end)
    iv = sorted((t(a), t(b)) for a, b in (x.split("-") for x in busy_a + busy_b))
    out, cur = [], lo
    for s, e in iv:
        s, e = max(s, lo), min(e, hi)
        if s >= e: continue
        if s > cur and s - cur >= minutes: out.append(f"{f(cur)}-{f(s)}")
        cur = max(cur, e)
    if hi - cur >= minutes and hi > cur: out.append(f"{f(cur)}-{f(hi)}")
    return out
''', ['(["09:00-10:00"], ["10:30-11:00"], "09:00", "12:00", 30)', '([], [], "09:00", "17:00", 60)',
      '(["08:00-09:30"], ["16:30-18:00"], "09:00", "17:00", 60)', '(["09:00-10:00", "09:30-11:00"], [], "09:00", "12:00", 30)',
      '(["10:00-10:30"], ["10:30-11:00"], "10:00", "12:00", 60)', '(["09:00-09:59"], [], "09:00", "10:00", 1)',
      '(["11:00-12:00"], ["09:00-10:00"], "09:00", "12:00", 61)', '(["00:00-23:59"], [], "09:00", "17:00", 1)',
      '(["12:00-13:00"], ["12:30-14:00"], "11:00", "15:00", 60)'], 3)

task("unique_prefixes", '''
def unique_prefixes(words: list[str]) -> list[str]:
    """For each word, return its shortest prefix that is not a prefix of any OTHER word in the
    list (other positions, including identical words). If no such prefix exists, return the
    whole word.
    Example: unique_prefixes(["dog", "cat", "apple", "apricot"]) == ["d", "c", "app", "apr"]
    """
''', '''
def unique_prefixes(words):
    out = []
    for i, w in enumerate(words):
        for k in range(1, len(w) + 1):
            p = w[:k]
            if not any(o.startswith(p) for j, o in enumerate(words) if j != i):
                out.append(p); break
        else:
            out.append(w)
    return out
''', ['(["dog", "cat", "apple", "apricot"])', '(["a", "ab"])', '(["zebra", "zebra"])', '([])', '(["", "a"])',
      '(["abc", "abd", "ab"])', '(["solo"])', '(["car", "cart", "cat"])'], 3)
