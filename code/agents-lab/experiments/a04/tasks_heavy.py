# 9 heavy, many-rule tasks. Executed inside tasks.py (uses its task() helper).

task("rrule_dates", r'''
def rrule_dates(start: str, rule: str) -> list[str]:
    """List the dates of a recurrence rule (a small subset of iCalendar RRULE) as "YYYY-MM-DD".
    - start is a date "YYYY-MM-DD". rule is ";"-separated KEY=VALUE parts in any order:
      FREQ (DAILY, WEEKLY or MONTHLY; required), INTERVAL (positive integer, default 1),
      COUNT (positive integer), UNTIL (a date, inclusive), BYDAY (comma-separated MO, TU, WE,
      TH, FR, SA, SU), BYMONTHDAY (comma-separated non-zero integers from -31 to 31; only used
      by MONTHLY). A missing FREQ, another FREQ value, or any other key raises ValueError.
    - Candidate dates:
      DAILY: start, start + INTERVAL days, start + 2*INTERVAL days, ...
      WEEKLY: weeks run Monday to Sunday; use every INTERVAL-th week, counting from the week
        that contains start. In each such week the candidates are the BYDAY days, or start's
        weekday if BYDAY is absent.
      MONTHLY: use every INTERVAL-th month, counting from start's month. In each such month
        the candidates are the BYMONTHDAY days (negative values count from the month's end,
        -1 is the last day; days that do not exist in that month are skipped); if BYMONTHDAY is
        absent, all days of the month when BYDAY is given, otherwise start's day of the month
        (skipped in months that lack it).
    - BYDAY also filters DAILY and MONTHLY candidates (keep only those weekdays).
    - Occurrences are the candidates on or after start, in date order. Stop after COUNT
      occurrences, or after UNTIL, whichever comes first; with neither, return the first 10.
      Never consider dates more than 3660 days after start.
    Example: rrule_dates("2026-10-24", "FREQ=WEEKLY;BYDAY=MO,FR;COUNT=3") == ["2026-10-26", "2026-10-30", "2026-11-02"]
    """
''', r'''
from datetime import date, timedelta
import calendar
DAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
def rrule_dates(start, rule):
    s = date.fromisoformat(start)
    p = {}
    for part in rule.split(";"):
        k, v = part.split("=", 1)
        if k not in ("FREQ", "INTERVAL", "COUNT", "UNTIL", "BYDAY", "BYMONTHDAY"): raise ValueError(k)
        p[k] = v
    if p.get("FREQ") not in ("DAILY", "WEEKLY", "MONTHLY"): raise ValueError("FREQ")
    freq = p["FREQ"]; n = int(p.get("INTERVAL", 1))
    count = int(p["COUNT"]) if "COUNT" in p else (None if "UNTIL" in p else 10)
    until = date.fromisoformat(p["UNTIL"]) if "UNTIL" in p else None
    byday = {DAYS.index(d) for d in p["BYDAY"].split(",")} if "BYDAY" in p else None
    bymd = [int(x) for x in p["BYMONTHDAY"].split(",")] if "BYMONTHDAY" in p else None
    limit = s + timedelta(days=3660)
    def cands():
        if freq == "DAILY":
            d = s
            while d <= limit:
                if byday is None or d.weekday() in byday: yield d
                d += timedelta(days=n)
        elif freq == "WEEKLY":
            wk = s - timedelta(days=s.weekday())
            while wk <= limit:
                for wd in (sorted(byday) if byday is not None else [s.weekday()]):
                    yield wk + timedelta(days=wd)
                wk += timedelta(weeks=n)
        else:
            y, m = s.year, s.month
            while date(y, m, 1) <= limit:
                last = calendar.monthrange(y, m)[1]
                if bymd is not None:
                    ds = sorted({x if x > 0 else last + 1 + x for x in bymd} & set(range(1, last + 1)))
                elif byday is not None:
                    ds = range(1, last + 1)
                else:
                    ds = [s.day] if s.day <= last else []
                for dd in ds:
                    d = date(y, m, dd)
                    if byday is None or d.weekday() in byday: yield d
                m += n
                while m > 12: m -= 12; y += 1
    out = []
    for d in cands():
        if d < s: continue
        if d > limit or (until and d > until): break
        out.append(d.isoformat())
        if count is not None and len(out) >= count: break
    return out
''', [r'("2026-10-24", "FREQ=WEEKLY;BYDAY=MO,FR;COUNT=3")', r'("2026-01-31", "FREQ=MONTHLY;COUNT=4")',
      r'("2026-10-24", "FREQ=DAILY;INTERVAL=2;UNTIL=2026-10-30")', r'("2026-10-24", "COUNT=2")',
      r'("2026-10-24", "FREQ=MONTHLY;BYMONTHDAY=1,-1;COUNT=4")', r'("2026-10-28", "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE;COUNT=4")',
      r'("2026-10-24", "FREQ=MONTHLY;BYDAY=SU;COUNT=3")', r'("2026-10-24", "FREQ=DAILY;BYDAY=SA,SU;COUNT=3")',
      r'("2026-10-24", "FREQ=WEEKLY")', r'("2026-10-24", "FREQ=YEARLY;COUNT=1")', r'("2026-10-24", "FREQ=DAILY;FOO=1")',
      r'("2024-02-29", "FREQ=MONTHLY;INTERVAL=12;COUNT=2")', r'("2026-10-24", "FREQ=MONTHLY;BYMONTHDAY=31;BYDAY=FR;UNTIL=2028-12-31")',
      r'("2026-10-24", "FREQ=WEEKLY;BYDAY=FR,MO;UNTIL=2026-11-03")', r'("2026-10-24", "INTERVAL=3;FREQ=DAILY;COUNT=2")',
      r'("2026-10-24", "FREQ=MONTHLY;BYMONTHDAY=-31,30;COUNT=3")'], 4)

task("tokenize", r'''
def tokenize(src: str) -> list[tuple]:
    """Tokenize source code of a small language into (kind, value) tuples.
    - Whitespace is skipped. "//" starts a comment to the end of the line. "/*" starts a block
      comment ending at the matching "*/"; block comments NEST ("/* a /* b */ c */" is one
      comment). An unterminated block comment raises ValueError.
    - Identifiers: an ASCII letter or "_" followed by ASCII letters, digits or "_". The keywords
      if, else, while, return give ("KW", word); other identifiers give ("ID", word).
    - Numbers start with a digit. "0x" or "0X" followed by one or more hex digits is ("INT",
      value). Otherwise: digits, optionally "." and digits (digits are required on both sides
      of the "."), optionally an exponent "e" or "E", an optional sign and digits. With a "." or
      an exponent it is ("FLOAT", float value), else ("INT", int value). Single underscores may
      separate digits in the integer and fraction parts ("1_000"). A number whose integer part
      has a leading zero followed by more digits ("007"), a misplaced underscore ("1__0", "1_"),
      or a number directly followed by a letter, digit or "_" raises ValueError.
    - Strings are double-quoted: ("STR", decoded text). Escapes: \n, \t, \\, \" and \u{X}
      with 1 to 6 hex digits (a code point up to 10FFFF). Any other escape, a raw newline
      inside the string, or a missing closing quote raises ValueError.
    - Operators, longest match first: == != <= >= && || -> + - * / % < > = ! ( ) { } [ ] , ; .
      give ("OP", text). Any other character raises ValueError.
    Example: tokenize("x = 1_000;") == [("ID", "x"), ("OP", "="), ("INT", 1000), ("OP", ";")]
    """
''', r'''
import re
KW = {"if", "else", "while", "return"}
OPS = ["==", "!=", "<=", ">=", "&&", "||", "->", "+", "-", "*", "/", "%", "<", ">", "=", "!",
       "(", ")", "{", "}", "[", "]", ",", ";", "."]
NUM = re.compile(r"[0-9]+(?:_[0-9]+)*(?:\.[0-9]+(?:_[0-9]+)*)?(?:[eE][+-]?[0-9]+)?")
def tokenize(src):
    out, i, n = [], 0, len(src)
    def word_char(c): return c == "_" or (c.isascii() and c.isalnum())
    while i < n:
        c = src[i]
        if c.isspace(): i += 1; continue
        if src.startswith("//", i):
            j = src.find("\n", i); i = n if j == -1 else j; continue
        if src.startswith("/*", i):
            depth, i = 1, i + 2
            while depth:
                if i >= n: raise ValueError("unterminated comment")
                if src.startswith("/*", i): depth += 1; i += 2
                elif src.startswith("*/", i): depth -= 1; i += 2
                else: i += 1
            continue
        if c == "_" or (c.isascii() and c.isalpha()):
            j = i
            while j < n and word_char(src[j]): j += 1
            w = src[i:j]; out.append(("KW" if w in KW else "ID", w)); i = j; continue
        if c.isascii() and c.isdigit():
            if src[i:i + 2] in ("0x", "0X"):
                j = i + 2
                while j < n and src[j] in "0123456789abcdefABCDEF": j += 1
                if j == i + 2: raise ValueError("bad hex")
                if j < n and word_char(src[j]): raise ValueError("bad number")
                out.append(("INT", int(src[i + 2:j], 16))); i = j; continue
            m = NUM.match(src, i); text = m.group(); j = m.end()
            if j < n and word_char(src[j]): raise ValueError("bad number")
            intpart = re.match(r"[0-9_]+", text).group()
            if len(intpart.replace("_", "")) > 1 and intpart[0] == "0": raise ValueError("leading zero")
            clean = text.replace("_", "")
            if "." in text or "e" in text or "E" in text: out.append(("FLOAT", float(clean)))
            else: out.append(("INT", int(clean)))
            i = j; continue
        if c == '"':
            j, buf = i + 1, []
            while True:
                if j >= n or src[j] == "\n": raise ValueError("unterminated string")
                ch = src[j]
                if ch == '"': j += 1; break
                if ch == "\\":
                    e = src[j + 1] if j + 1 < n else ""
                    if e in ("n", "t", "\\", '"'):
                        buf.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}[e]); j += 2
                    elif e == "u":
                        m = re.compile(r"u\{([0-9a-fA-F]{1,6})\}").match(src, j + 1)
                        if not m or int(m.group(1), 16) > 0x10FFFF: raise ValueError("bad escape")
                        buf.append(chr(int(m.group(1), 16))); j = m.end()
                    else:
                        raise ValueError("bad escape")
                    continue
                buf.append(ch); j += 1
            out.append(("STR", "".join(buf))); i = j; continue
        for op in OPS:
            if src.startswith(op, i):
                out.append(("OP", op)); i += len(op); break
        else:
            raise ValueError(f"bad character {c!r}")
    return out
''', [r'("x = 1_000;")', r'''('if (a>=b) return "hi\\n";')''', r'("/* a /* b */ c */ y")', r'("3.14 1e3 2.5E-2")',
      r'("/* open")', r'("0x1F + 0")', r'("007")', r'("1__0")', r'''('"\\u{1F600}"')''', r'''('"bad \\q"')''',
      r'("a->b && !c || d != e")', r'("1.foo")', r'("12abc")', r'''('"line\nbreak"')''', r'("_x9 // note\nelse")',
      r'("a @ b")', r'("1e")', r'''('"')''', r'("0.5 0 1_0.2_5")', r'("x/*c*/-1")'], 5)

task("bowling_validate", r'''
def bowling_validate(rolls: list[int]) -> int:
    """Score a ten-pin bowling game given the pins knocked down by each roll, validating it.
    - A game has 10 frames. In frames 1-9 a strike (10) is a single roll; otherwise the frame has
      two rolls whose sum is at most 10 (a spare if exactly 10).
    - Frame 10: a strike earns two bonus rolls; if the first bonus roll is not a strike, the two
      bonus rolls together are at most 10. A spare earns one bonus roll. Otherwise no bonus.
    - Score: each frame scores its pins; a strike adds the next two rolls, a spare the next roll.
      Bonus rolls of frame 10 only count as bonuses.
    - Raise ValueError for any roll outside 0-10, too many pins in a frame, an incomplete game,
      or extra rolls after the game is over.
    Example: bowling_validate([10] * 12) == 300
    """
''', r'''
def bowling_validate(rolls):
    if any(r < 0 or r > 10 for r in rolls): raise ValueError("bad roll")
    def get(k):
        if k >= len(rolls): raise ValueError("incomplete")
        return rolls[k]
    i, score = 0, 0
    for _ in range(9):
        a = get(i)
        if a == 10:
            score += 10 + get(i + 1) + get(i + 2); i += 1
        else:
            b = get(i + 1)
            if a + b > 10: raise ValueError("frame")
            score += a + b + (get(i + 2) if a + b == 10 else 0); i += 2
    a = get(i)
    if a == 10:
        b, c = get(i + 1), get(i + 2)
        if b != 10 and b + c > 10: raise ValueError("bonus")
        score += 10 + b + c; i += 3
    else:
        b = get(i + 1)
        if a + b > 10: raise ValueError("frame")
        if a + b == 10:
            score += 10 + get(i + 2); i += 3
        else:
            score += a + b; i += 2
    if i != len(rolls): raise ValueError("extra rolls")
    return score
''', [r"([10] * 12)", r"([0] * 20)", r"([10] * 11)", r"([5, 6] + [0] * 18)", r"([0] * 21)", r"([0] * 18 + [10, 5, 6])",
      r"([0] * 18 + [10, 10, 6])", r"([0] * 18 + [7, 3, 10])", r"([0] * 18 + [7, 3])", r"([0] * 18 + [3, 4, 5])",
      r"([11] + [0] * 19)", r"([10] * 9 + [9, 1, 10])", r"([0] * 18 + [10, 0, 10])", r"([3] * 20)", r"([10, 5, 5] + [0] * 16)",
      r"([0] * 18 + [10, 7, 3])", r"([-1, 1] + [0] * 18)"], 4)

task("render", r'''
def render(template: str, ctx: dict) -> str:
    """Render a small template language.
    - Text outside tags is copied as is, except that a backslash immediately before "{" is
      dropped and that "{" is plain text (so backslash + "{{" outputs "{{" literally).
    - "{{ expr }}" outputs a value; spaces inside the braces are optional. expr is a name
      followed by optional filters: "name|filter|filter" (spaces around "|" are ignored).
      A name is letters, digits, "_" and "."; "user.name" looks up nested dicts. Inside a for
      loop, "loop.index" is the 1-based iteration number of the innermost loop. A name that
      cannot be resolved is missing.
    - A missing value or None prints as ""; any other value prints as str(value). The printed
      text is HTML-escaped (& < > " ' become &amp; &lt; &gt; &quot; &#39;) unless the safe
      filter was applied.
    - Filters, applied left to right: upper, lower, trim (strip whitespace) act on str(value)
      and leave a missing value missing; length gives len(value), or 0 if missing;
      default:TEXT replaces a missing, None or "" value with the literal TEXT (up to the next
      "|" or the end); safe disables escaping. An unknown filter raises ValueError.
    - "{% if name %}A{% else %}B{% endif %}" (else optional) renders A if name resolves to a
      truthy value (missing is false), else B. "{% for x in name %}...{% endfor %}" renders the
      body once per item, with x bound to the item (shadowing outer names); a missing name
      renders nothing. Blocks nest; spaces inside "{% %}" are flexible.
    - An unclosed "{{" or "{%", or a mismatched, unknown or unclosed block tag raises ValueError.
    Example: render("Hi {{ name|upper }}!", {"name": "<ann>"}) == "Hi &lt;ANN&gt;!"
    """
''', r'''
import re
MISSING = object()
def render(template, ctx):
    toks, i, buf = [], 0, []
    while i < len(template):
        if template.startswith("\\{", i):
            buf.append("{"); i += 2; continue
        if template.startswith("{{", i) or template.startswith("{%", i):
            close = "}}" if template[i + 1] == "{" else "%}"
            j = template.find(close, i + 2)
            if j == -1: raise ValueError("unclosed tag")
            if buf: toks.append(("text", "".join(buf))); buf = []
            toks.append(("var" if close == "}}" else "tag", template[i + 2:j].strip())); i = j + 2; continue
        buf.append(template[i]); i += 1
    if buf: toks.append(("text", "".join(buf)))
    root = []; stack = [["root", root, None]]
    for kind, val in toks:
        if kind != "tag":
            stack[-1][1].append((kind, val)); continue
        m = re.fullmatch(r"if\s+([\w.]+)", val)
        if m:
            node = {"t": "if", "name": m[1], "a": [], "b": []}
            stack[-1][1].append(node); stack.append(["if", node["a"], node]); continue
        if val == "else":
            if stack[-1][0] != "if": raise ValueError("else")
            stack[-1] = ["else", stack[-1][2]["b"], stack[-1][2]]; continue
        if val == "endif":
            if stack[-1][0] not in ("if", "else"): raise ValueError("endif")
            stack.pop(); continue
        m = re.fullmatch(r"for\s+(\w+)\s+in\s+([\w.]+)", val)
        if m:
            node = {"t": "for", "var": m[1], "name": m[2], "body": []}
            stack[-1][1].append(node); stack.append(["for", node["body"], node]); continue
        if val == "endfor":
            if stack[-1][0] != "for": raise ValueError("endfor")
            stack.pop(); continue
        raise ValueError("unknown tag")
    if len(stack) != 1: raise ValueError("unclosed block")
    def resolve(name, scope):
        parts = name.split(".")
        cur = scope.get(parts[0], MISSING)
        for p in parts[1:]:
            if isinstance(cur, dict) and p in cur: cur = cur[p]
            else: return MISSING
        return cur
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    def var(expr, scope):
        parts = [p.strip() for p in expr.split("|")]
        v, safe = resolve(parts[0], scope), False
        for f in parts[1:]:
            if f in ("upper", "lower", "trim"):
                if v is not MISSING: v = {"upper": str.upper, "lower": str.lower, "trim": str.strip}[f](str(v))
            elif f == "length": v = 0 if v is MISSING else len(v)
            elif f.startswith("default:"):
                if v is MISSING or v is None or (isinstance(v, str) and v == ""): v = f[len("default:"):]
            elif f == "safe": safe = True
            else: raise ValueError("unknown filter")
        s = "" if v is MISSING or v is None else str(v)
        return s if safe else esc(s)
    def run(nodes, scope):
        out = []
        for nd in nodes:
            if isinstance(nd, tuple):
                out.append(nd[1] if nd[0] == "text" else var(nd[1], scope))
            elif nd["t"] == "if":
                v = resolve(nd["name"], scope)
                out.append(run(nd["a"] if (v is not MISSING and v) else nd["b"], scope))
            else:
                seq = resolve(nd["name"], scope)
                if seq is MISSING: continue
                for k, item in enumerate(seq, 1):
                    out.append(run(nd["body"], {**scope, nd["var"]: item, "loop": {"index": k}}))
        return "".join(out)
    return run(root, dict(ctx))
''', [r'("Hi {{ name|upper }}!", {"name": "<ann>"})', r'("{% if admin %}A{% else %}U{% endif %}", {})',
      r'("{% for x in xs %}{{loop.index}}:{{x}} {% endfor %}", {"xs": ["a", "b"]})', r'("\\{{ x }}", {"x": 1})',
      r'("{{ user.name|default:anon }}", {"user": {}})', r'("{{ q }}", {"q": "it\'s \"ok\""})', r'("{{ q|safe }}", {"q": "<b>"})',
      r'("{{ xs|length }}", {"xs": [1, 2, 3]})', r'("{{ missing|length }}", {})', r'("{{ x|bogus }}", {"x": 1})',
      r'("{% if a %}{% if b %}AB{% endif %}{% endif %}", {"a": 1, "b": 0})', r'("{% for x in xs %}{% endif %}", {"xs": []})',
      r'("{% if a %}x", {"a": 1})', r'("{{ n }}", {"n": None})', r'("{{ s | trim | upper }}", {"s": "  hi "})',
      r'("{{ e|default:none }}", {"e": ""})', r'("{% for x in xs %}{{ x.v }}{% endfor %}", {"xs": [{"v": 1}, {"w": 2}]})',
      r'("{{ x }}{{ x|upper }}", {"x": 0})', r'("{%for i in xs%}{%if i%}{{i}}{%else%}-{%endif%}{%endfor%}", {"xs": [0, 2, ""]})',
      r'("{{ x }", {})', r'("A & B {{ x }}", {"x": "&"})',
      r'("{% for x in xs %}{{ loop.index }}{% for y in ys %}{{ loop.index }}{% endfor %}{{ loop.index }}{% endfor %}", {"xs": [1, 2], "ys": [1, 2]})',
      r'("{{ missing|upper|default:X }}", {})', r'("{% for x in nope %}x{% endfor %}{{ x }}", {"x": "outer"})',
      r'("{% if a %}1{% else %}2{% else %}3{% endif %}", {})'], 5)

task("iso_duration", r'''
def iso_duration(s: str) -> float:
    """Convert an ISO 8601 duration to a number of seconds, returned as a float.
    - Format: "P", then optional date components nY, nM, nW, nD in that order, then optionally
      "T" followed by time components nH, nM, nS in that order. Each component appears at most
      once. M before T means months; M after T means minutes.
    - Values: Y = 365 days, M (months) = 30 days, W = 7 days, D = 86400 s, H = 3600 s,
      M (minutes) = 60 s, S = 1 s.
    - n is one or more ASCII digits, optionally with a fraction written with "." or ","
      ("0.5" or "0,5", digits on both sides). Only the last component may have a fraction.
    - There must be at least one component, and a "T" must be followed by at least one time
      component. Anything else (lowercase letters, spaces, signs, empty components) raises
      ValueError.
    Example: iso_duration("PT1H30M") == 5400.0
    """
''', r'''
import re
def iso_duration(s):
    num = r"([0-9]+(?:[.,][0-9]+)?)"
    m = re.fullmatch(rf"P(?:{num}Y)?(?:{num}M)?(?:{num}W)?(?:{num}D)?(?:T(?:{num}H)?(?:{num}M)?(?:{num}S)?)?", s)
    if not m or s.endswith("T"): raise ValueError(s)
    vals = m.groups()
    present = [i for i, v in enumerate(vals) if v is not None]
    if not present: raise ValueError(s)
    for i in present[:-1]:
        if not vals[i].isdigit(): raise ValueError(s)
    mult = [365 * 86400, 30 * 86400, 7 * 86400, 86400, 3600, 60, 1]
    return float(sum(float(vals[i].replace(",", ".")) * mult[i] for i in present))
''', [r'("PT1H30M")', r'("P1M")', r'("PT1M")', r'("P")', r'("P1DT")', r'("PT0,5S")', r'("P1Y2M")', r'("P1.5DT1H")',
      r'("p1d")', r'("P-1D")', r'("PT36H")', r'("P0D")', r'("P1DT2H3M4.5S")', r'("P1W2D")', r'("PT")', r'("P1H")',
      r'("PT1.5H")', r'("P1Y 2M")', r'("PT.5S")'], 4)

task("month_calendar", r'''
def month_calendar(year: int, month: int, first_weekday: int) -> str:
    """Lay out one month as text, like a wall calendar. first_weekday is 0 for weeks starting
    on Monday ... 6 for Sunday.
    - Line 1: the English month name and the year ("October 2026"), centered in 20 characters:
      left padding is (20 - length) // 2 spaces.
    - Line 2: two-letter weekday names (Mo Tu We Th Fr Sa Su) starting at first_weekday,
      separated by single spaces.
    - Then one line per week: each day number right-aligned in 2 characters, days separated by
      single spaces, with two spaces for each position before the 1st.
    - Strip trailing spaces from every line; join lines with "\n"; no trailing newline.
    Example: month_calendar(2026, 2, 0).split("\n")[2] == "                   1"
    """
''', r'''
import calendar
NAMES = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
def month_calendar(year, month, first_weekday):
    title = f"{calendar.month_name[month]} {year}"
    lines = [(" " * ((20 - len(title)) // 2) + title).rstrip()]
    lines.append(" ".join(NAMES[(first_weekday + k) % 7] for k in range(7)))
    first, ndays = calendar.monthrange(year, month)
    cells = ["  "] * ((first - first_weekday) % 7) + [f"{d:2d}" for d in range(1, ndays + 1)]
    for k in range(0, len(cells), 7):
        lines.append(" ".join(cells[k:k + 7]).rstrip())
    return "\n".join(lines)
''', [r"(2026, 2, 0)", r"(2026, 10, 0)", r"(2026, 10, 6)", r"(2026, 2, 6)", r"(2024, 2, 3)", r"(2026, 9, 2)",
      r"(2027, 5, 5)", r"(2025, 12, 1)"], 3)

task("http_range", r'''
def http_range(header: str, size: int) -> list[tuple[int, int]] | None:
    """Interpret an HTTP Range header for a resource of `size` bytes.
    - The header is "bytes=" (the unit "bytes" is case-insensitive) followed by comma-separated
      range specs. Spaces and tabs around specs are ignored and empty list elements are skipped.
      If the unit is not bytes, there are no specs, or any spec is malformed, return None
      (the header is ignored).
    - Specs: "a-b" (bytes a to b inclusive; if b < a the header is malformed), "a-" (from a to
      the end) and "-n" (the last n bytes). a, b and n are non-negative decimal integers.
    - A spec is satisfiable if it includes at least one existing byte: "a-b" and "a-" need
      a < size, "-n" needs n > 0 and size > 0. Clamp ends to size - 1; "-n" with n > size starts
      at 0. Unsatisfiable specs are dropped.
    - Return the satisfiable ranges as (start, end) tuples sorted by start, merging ranges that
      overlap or are adjacent. If the header is valid but no spec is satisfiable, raise ValueError.
    Example: http_range("bytes=0-499", 1000) == [(0, 499)]
    """
''', r'''
import re
def http_range(header, size):
    unit, sep, rest = header.partition("=")
    if not sep or unit.lower() != "bytes": return None
    specs = [x.strip(" \t") for x in rest.split(",")]
    specs = [x for x in specs if x]
    if not specs: return None
    rngs = []
    for sp in specs:
        m = re.fullmatch(r"([0-9]+)-([0-9]*)|-([0-9]+)", sp)
        if not m: return None
        if m[3] is not None:
            k = int(m[3])
            if k == 0 or size == 0: continue
            rngs.append((max(0, size - k), size - 1))
        else:
            a = int(m[1]); b = int(m[2]) if m[2] else None
            if b is not None and b < a: return None
            if a >= size: continue
            rngs.append((a, size - 1 if b is None else min(b, size - 1)))
    if not rngs: raise ValueError("unsatisfiable")
    rngs.sort(); out = [list(rngs[0])]
    for s, e in rngs[1:]:
        if s <= out[-1][1] + 1: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return [tuple(x) for x in out]
''', [r'("bytes=0-499", 1000)', r'("bytes=-500", 1000)', r'("bytes=9500-", 10000)', r'("items=0-1", 10)',
      r'("bytes=0-0,-1", 10000)', r'("bytes=500-600,601-999", 1000)', r'("bytes=5-1", 10)', r'("bytes=20-30", 10)',
      r'("bytes=-0", 10)', r'("bytes=0-5, 3-8 ,", 10)', r'("BYTES=0-1", 10)', r'("bytes=-20", 10)', r'("bytes=1-2-3", 10)',
      r'("bytes=", 10)', r'("bytes=0-1,x", 10)', r'("bytes=0-100", 0)', r'("bytes=9-", 10)', r'("bytes=20-30,5-1", 10)',
      r'("bytes=\t2-3\t,0-1", 10)'], 4)

task("render_table", r'''
def render_table(headers: list[str], rows: list[list], aligns: str | None = None) -> str:
    """Render rows as an ASCII grid table.
    - Cell text is str(value), except None, which is "". A cell's text may contain "\n": the
      row is then as many lines tall as its tallest cell, and shorter cells get empty lines
      below their text.
    - Column width is the longest line among the header and all cells of that column.
    - Alignment per column: if aligns is given, its i-th character is "l", "r" or "c" for
      column i. Otherwise a column is right-aligned when every non-None value in it is an int
      or float (bool does not count as a number) and there is at least one such value, and
      left-aligned otherwise. Headers use the same alignment as their column. "c" centers with
      any odd extra space on the right.
    - Layout: a border line "+" + "+".join("-" * (width + 2) for each column) + "+"; the header
      line; a line like the border but with "=" instead of "-"; each row's lines; the border
      again. Text lines are "| " + " | ".join(padded cells) + " |".
    - Lines are joined with "\n"; no trailing newline. An empty rows list still shows the
      header between the border and "=" lines and the closing border.
    Example: render_table(["a"], [[1]]) == "+---+\n| a |\n+===+\n| 1 |\n+---+"
    """
''', r'''
def render_table(headers, rows, aligns=None):
    ncol = len(headers)
    def txt(v): return "" if v is None else str(v)
    def isnum(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
    cols = [[r[i] for r in rows] for i in range(ncol)]
    widths = [max([len(l) for l in headers[i].split("\n")] + [len(l) for v in cols[i] for l in txt(v).split("\n")]) for i in range(ncol)]
    if aligns is None:
        al = []
        for i in range(ncol):
            vals = [v for v in cols[i] if v is not None]
            al.append("r" if vals and all(isnum(v) for v in vals) else "l")
    else:
        al = list(aligns)
    def pad(s, w, a):
        if a == "r": return s.rjust(w)
        if a == "c":
            left = (w - len(s)) // 2
            return " " * left + s + " " * (w - len(s) - left)
        return s.ljust(w)
    def lines_of(cells):
        parts = [txt(c).split("\n") for c in cells]
        h = max(len(p) for p in parts)
        out = []
        for k in range(h):
            out.append("| " + " | ".join(pad(p[k] if k < len(p) else "", widths[i], al[i]) for i, p in enumerate(parts)) + " |")
        return out
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [border] + lines_of(headers) + [border.replace("-", "=")]
    for r in rows: out += lines_of(r)
    out.append(border)
    return "\n".join(out)
''', [r'(["a"], [[1]])', r'(["name", "qty"], [["apple", 3], ["kiwi", 12]])', r'(["x"], [])', r'(["k", "v"], [["a", None], ["bb", 2.5]])',
      r'(["flag"], [[True], [False]])', r'(["n"], [[None], [7]])', r'(["c"], [["ab"], ["abcde"]], "c")',
      r'(["id", "note"], [[1, "two\nlines"], [22, "x"]])', r'(["h"], [["a"]], "r")', r'(["mixed"], [[1], ["a"]])',
      r'(["n"], [[None]])'], 3)

task("lfu_cache", r'''
def lfu_cache(capacity: int, ops: list[tuple]) -> list[int]:
    """Simulate a least-frequently-used cache and return the result of every "get", in order.
    - ops are ("put", key, value) or ("get", key). get returns the value, or -1 if absent.
    - Each key has a use count: a put that inserts a key sets it to 1; a get of a present key
      and a put that updates a present key each add 1. A put that updates never evicts.
    - When a put inserts a new key and the cache already holds `capacity` keys, first evict the
      key with the smallest use count; among ties, evict the one used least recently (a get or
      put of a key counts as a use). With capacity 0 nothing is ever stored.
    Example: lfu_cache(1, [("put", 1, 1), ("get", 1), ("put", 2, 2), ("get", 1), ("get", 2)]) == [1, -1, 2]
    """
''', r'''
def lfu_cache(capacity, ops):
    val, cnt, last, out, clock = {}, {}, {}, [], 0
    for op in ops:
        clock += 1
        if op[0] == "get":
            k = op[1]
            if k in val:
                cnt[k] += 1; last[k] = clock; out.append(val[k])
            else:
                out.append(-1)
        else:
            _, k, v = op
            if capacity <= 0: continue
            if k in val:
                val[k] = v; cnt[k] += 1; last[k] = clock; continue
            if len(val) >= capacity:
                victim = min(val, key=lambda x: (cnt[x], last[x]))
                del val[victim], cnt[victim], last[victim]
            val[k] = v; cnt[k] = 1; last[k] = clock
    return out
''', [r'(1, [("put", 1, 1), ("get", 1), ("put", 2, 2), ("get", 1), ("get", 2)])',
      r'(2, [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 3), ("put", 4, 4), ("get", 1), ("get", 3), ("get", 4)])',
      r'(0, [("put", 1, 1), ("get", 1)])', r'(2, [("put", 1, 1), ("put", 2, 2), ("put", 1, 10), ("put", 3, 3), ("get", 1), ("get", 2), ("get", 3)])',
      r'(2, [("put", 1, 1), ("put", 2, 2), ("get", 2), ("get", 1), ("put", 3, 3), ("get", 1), ("get", 2), ("get", 3)])',
      r'(1, [("get", 5), ("put", 5, 0), ("get", 5)])', r'(3, [("put", "a", 1), ("put", "b", 2), ("put", "c", 3), ("get", "a"), ("get", "a"), ("get", "b"), ("put", "d", 4), ("get", "c"), ("get", "d"), ("put", "e", 5), ("get", "b"), ("get", "d")])',
      r'(2, [("put", 1, 1), ("put", 1, 2), ("put", 1, 3), ("put", 2, 1), ("put", 3, 1), ("get", 1), ("get", 2), ("get", 3)])'], 3)
