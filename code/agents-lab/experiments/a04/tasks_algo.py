# 7 algorithmic tasks where off-by-one and counting mistakes are common.
# Executed inside tasks.py (uses its task() helper). References are brute-force checked in check_algo.py.

task("even_heavy_count", r'''
def even_heavy_count(lo: int, hi: int) -> int:
    """Count the integers x with lo <= x <= hi (0 <= lo <= hi <= 10**18) whose decimal
    representation (no leading zeros; 0 is written "0") has strictly more even digits than odd
    digits. 0 is an even digit. Must be fast for the full range.
    Example: even_heavy_count(1, 30) == 9
    """
''', r'''
from functools import lru_cache
def even_heavy_count(lo, hi):
    def f(n):  # count in [0, n]
        if n < 0: return 0
        ds = list(map(int, str(n)))
        @lru_cache(None)
        def go(i, diff, started, tight):
            if i == len(ds):
                if not started: return 1  # the number 0: one even digit
                return 1 if diff > 0 else 0
            lim = ds[i] if tight else 9
            total = 0
            for d in range(lim + 1):
                st = started or d != 0
                nd = diff + ((1 if d % 2 == 0 else -1) if st else 0)
                total += go(i + 1, nd, st, tight and d == lim)
            return total
        return go(0, 0, False, True)
    return f(hi) - f(lo - 1)
''', ["(1, 30)", "(0, 0)", "(0, 9)", "(10, 99)", "(100, 999)", "(1, 10**18)", "(123456789, 987654321)",
      "(10**17, 10**17 + 12345)", "(5, 5)", "(200, 200)", "(201, 201)", "(999999999999, 1000000000000)"], 4)

task("calm_numbers", r'''
def calm_numbers(n: int) -> int:
    """Count the integers x with 1 <= x <= n (0 <= n <= 10**18) such that no two adjacent
    decimal digits of x are equal and the sum of the digits of x is divisible by 7.
    Must be fast for n up to 10**18.
    Example: calm_numbers(100) == 12
    """
''', r'''
from functools import lru_cache
def calm_numbers(n):
    if n <= 0: return 0
    ds = list(map(int, str(n)))
    @lru_cache(None)
    def go(i, last, mod, tight):
        if i == len(ds):
            return 1 if last != -1 and mod == 0 else 0
        lim = ds[i] if tight else 9
        total = 0
        for d in range(lim + 1):
            if last == -1 and d == 0:
                total += go(i + 1, -1, 0, tight and d == lim)
            elif d != last:
                total += go(i + 1, d, (mod + d) % 7, tight and d == lim)
        return total
    return go(0, -1, 0, True)
''', ["(100)", "(0)", "(7)", "(70)", "(1000)", "(10**18)", "(123456789)", "(77)", "(16)", "(999999999999999999)", "(61)"], 4)

task("kth_string", r'''
def kth_string(n: int, k: int) -> str:
    """Consider all strings of length n over the alphabet "abc" that have no two equal adjacent
    characters and do not contain "cab" as a substring. Return the k-th of them (k = 1 is the
    first) in lexicographic order, or "" if there are fewer than k such strings.
    1 <= n <= 60 and 1 <= k <= 10**18, so do not enumerate the strings.
    Example: kth_string(3, 1) == "aba"
    """
''', r'''
from functools import lru_cache
def kth_string(n, k):
    @lru_cache(None)
    def count(rem, tail):  # tail: last two characters placed (or fewer)
        if rem == 0: return 1
        total = 0
        for c in "abc":
            if tail and tail[-1] == c: continue
            if (tail + c)[-3:] == "cab": continue
            total += count(rem - 1, (tail + c)[-2:])
        return total
    if count(n, "") < k: return ""
    out, tail = [], ""
    for i in range(n):
        for c in "abc":
            if tail and tail[-1] == c: continue
            if (tail + c)[-3:] == "cab": continue
            m = count(n - i - 1, (tail + c)[-2:])
            if k <= m:
                out.append(c); tail = (tail + c)[-2:]; break
            k -= m
    return "".join(out)
''', ["(3, 1)", "(1, 3)", "(1, 4)", "(2, 6)", "(3, 11)", "(4, 1)", "(5, 17)", "(60, 10**18)", "(60, 1)",
      "(40, 12345678901)", "(10, 300)", "(3, 12)", "(60, 4193837582544676)", "(60, 4193837582544677)"], 4)

task("prime_josephus", r'''
def prime_josephus(n: int) -> list[int]:
    """People numbered 1..n stand in a circle in order. Eliminate them one at a time: for the
    i-th elimination (i = 1, 2, ...) let p be the i-th prime (2, 3, 5, 7, ...). Count p people
    around the circle, starting with (and counting) the current person; the p-th person
    counted is eliminated, and the person after them becomes the current person. The first
    current person is 1. Return the order in which people are eliminated (all n of them).
    n <= 3000.
    Example: prime_josephus(5) == [2, 5, 3, 4, 1]
    """
''', r'''
def prime_josephus(n):
    primes, c = [], 2
    while len(primes) < n:
        if all(c % p for p in primes if p * p <= c): primes.append(c)
        c += 1
    people, idx, out = list(range(1, n + 1)), 0, []
    for i in range(n):
        idx = (idx + primes[i] - 1) % len(people)
        out.append(people.pop(idx))
        if people: idx %= len(people)
    return out
''', ["(5)", "(1)", "(2)", "(3)", "(10)", "(7)", "(3000)", "(100)"], 3)

task("count_true", r'''
def count_true(expr: str) -> int:
    """Count the assignments of the variables in a boolean expression that make it true.
    - Tokens: variables (single lowercase letters), the constants 0 and 1, parentheses, and the
      operators ! (NOT, prefix), & (AND), ^ (XOR), | (OR) and -> (IMPLIES). Spaces may appear
      between tokens.
    - Precedence from highest to lowest: !, &, ^, |, ->. The operators &, ^ and | are
      left-associative; -> is right-associative (a -> b -> c means a -> (b -> c)).
    - With k distinct variables there are 2**k assignments (one if there are no variables).
    Example: count_true("a & !b") == 1
    """
''', r'''
import itertools, re
def count_true(expr):
    toks = re.findall(r"->|[a-z01()!&^|]", expr.replace(" ", ""))
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def take(): pos[0] += 1; return toks[pos[0] - 1]
    def imp():
        a = orr()
        if peek() == "->":
            take(); b = imp(); return lambda e, a=a, b=b: (not a(e)) or b(e)
        return a
    def binop(sub, op, f):
        def parse():
            a = sub()
            while peek() == op:
                take(); b = sub(); a = (lambda a, b: lambda e: f(a(e), b(e)))(a, b)
            return a
        return parse
    def un():
        if peek() == "!":
            take(); a = un(); return lambda e, a=a: not a(e)
        t = take()
        if t == "(":
            a = imp(); take(); return a
        if t in "01": return lambda e, v=(t == "1"): v
        return lambda e, t=t: e[t]
    andd = binop(un, "&", lambda x, y: x and y)
    xorr = binop(andd, "^", lambda x, y: x != y)
    orr = binop(xorr, "|", lambda x, y: x or y)
    f = imp()
    vs = sorted(set(t for t in toks if t.isalpha()))
    return sum(1 for bits in itertools.product([False, True], repeat=len(vs)) if f(dict(zip(vs, bits))))
''', ['("a & !b")', '("a | b ^ c")', '("a -> b -> c")', '("(a -> b) -> c")', '("!a & b | c")', '("1")', '("0 -> 0")',
      '("a ^ b ^ c ^ d")', '("!!a")', '("a & b ^ a | b")', '("(a|b) & (c|d) -> e ^ f")', '("x -> y & !x -> z")',
      '("a^b&c")', '("! (a | b) | c & d -> a ^ b")'], 4)

task("best_fraction", r'''
def best_fraction(x: str, max_den: int) -> str:
    """Return the fraction p/q with 1 <= q <= max_den closest to the exact value of the
    decimal string x (like "3.14159" or "-0.333"), as "p/q" in lowest terms with the sign on p
    ("-1/3", "3/1"). If two fractions are equally close, return the one with the smaller q.
    max_den can be up to 10**12, so do not search all denominators.
    Example: best_fraction("3.14159", 100) == "311/99"
    """
''', r'''
from fractions import Fraction
def best_fraction(x, max_den):
    v = Fraction(x)
    if v.denominator <= max_den: return f"{v.numerator}/{v.denominator}"
    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = v.numerator, v.denominator
    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > max_den: break
        p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
        n, d = d, n - a * d
    k = (max_den - q0) // q1
    b1 = Fraction(p0 + k * p1, q0 + k * q1); b2 = Fraction(p1, q1)
    e1, e2 = abs(b1 - v), abs(b2 - v)
    best = b1 if (e1 < e2 or (e1 == e2 and b1.denominator < b2.denominator)) else b2
    return f"{best.numerator}/{best.denominator}"
''', ['("3.14159", 100)', '("0.5", 10)', '("-0.333", 10)', '("2", 5)', '("0.25", 3)', '("0.1", 1)',
      '("3.141592653589793", 1000000)', '("0.6", 2)', '("-2.4", 1)', '("0.0", 7)', '("1.7320508075688772", 10**12)',
      '("0.75", 2)', '("-0.15", 3)', '("123.456", 1000)'], 4)

task("expand_poly", r'''
def expand_poly(expr: str) -> str:
    """Expand a polynomial expression in x and return it in canonical form.
    - Input: the variable x, non-negative integer constants, +, -, * and ^, parentheses, unary
      minus, and spaces. ^ has the highest precedence and its right side is always a
      non-negative integer constant. Unary minus binds looser than ^ (-x^2 is -(x^2)) but
      tighter than * (2*-x is -2x). There is no implicit multiplication.
    - Output: non-zero terms in decreasing power, no spaces. A term is the coefficient followed
      by x^k; write x instead of x^1, leave out x^0, and leave out a coefficient of 1 (or -1,
      keeping the "-") on terms with x. Terms after the first are joined with "+" or "-".
      The zero polynomial is "0".
    Example: expand_poly("(x+1)^2 - 1") == "x^2+2x"
    """
''', r'''
import re
def expand_poly(expr):
    toks = re.findall(r"\d+|[x+\-*^()]", expr)
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def take(): pos[0] += 1; return toks[pos[0] - 1]
    def add(a, b):
        n = max(len(a), len(b)); return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]
    def mul(a, b):
        out = [0] * (len(a) + len(b) - 1)
        for i, u in enumerate(a):
            for j, w in enumerate(b): out[i + j] += u * w
        return out
    def neg(a): return [-c for c in a]
    def expr_():
        a = term()
        while peek() in ("+", "-"):
            op = take(); b = term(); a = add(a, b if op == "+" else neg(b))
        return a
    def term():
        a = unary()
        while peek() == "*":
            take(); a = mul(a, unary())
        return a
    def unary():
        if peek() == "-":
            take(); return neg(unary())
        return power()
    def power():
        b = atom()
        while peek() == "^":
            take(); e = int(take()); r = [1]
            for _ in range(e): r = mul(r, b)
            b = r
        return b
    def atom():
        t = take()
        if t == "(":
            a = expr_(); take(); return a
        if t == "x": return [0, 1]
        return [int(t)]
    c = expr_()
    terms = []
    for k in range(len(c) - 1, -1, -1):
        v = c[k]
        if v == 0: continue
        if k == 0: body = str(abs(v))
        else:
            xs = "x" if k == 1 else f"x^{k}"
            body = xs if abs(v) == 1 else f"{abs(v)}{xs}"
        sign = "-" if v < 0 else "+"
        terms.append((sign, body))
    if not terms: return "0"
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]: s += sign + body
    return s
''', ['("(x+1)^2 - 1")', '("(x+1)^3")', '("x-x")', '("-(x-1)*(x+1)")', '("2*x^0")', '("-x^2")', '("(2*x-3)^2-4*x^2")',
      '("x*x*x - x^3 + 5")', '("((x))")', '("(x+2)^0")', '("0*x")', '("3-5")', '("(x-1)^10")', '("--x")', '("2*-x")',
      '("-1 + x^2 - x^2")'], 4)
