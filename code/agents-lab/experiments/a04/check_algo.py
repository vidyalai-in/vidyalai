"""Brute-force checks of the algorithmic task references (small inputs)."""
import itertools, random, re, sys
from fractions import Fraction
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a04")
from tasks import TASKS
ns = {}
for t in TASKS:
    exec(t["ref"], ns)
random.seed(0)
# even_heavy_count
def eh(x):
    s = str(x); e = sum(c in "02468" for c in s); return e > len(s) - e
for _ in range(300):
    lo = random.randint(0, 3000); hi = lo + random.randint(0, 3000)
    assert ns["even_heavy_count"](lo, hi) == sum(eh(x) for x in range(lo, hi + 1)), (lo, hi)
# calm_numbers
def calm(x):
    s = str(x); return all(a != b for a, b in zip(s, s[1:])) and sum(map(int, s)) % 7 == 0
acc = 0
for n in range(0, 20001):
    if n and calm(n): acc += 1
    if n % 997 == 0 or n < 200: assert ns["calm_numbers"](n) == acc, n
# kth_string
for n in range(1, 9):
    good = sorted("".join(p) for p in itertools.product("abc", repeat=n)
                  if all(a != b for a, b in zip(p, p[1:])) and "cab" not in "".join(p))
    for k in range(1, len(good) + 3):
        assert ns["kth_string"](n, k) == (good[k - 1] if k <= len(good) else ""), (n, k)
# count_true vs Python eval
def py(expr):
    e = expr
    vs = sorted(set(re.findall(r"[a-z]", e)))
    return vs
for t in TASKS:
    if t["id"] == "count_true":
        pass
# best_fraction brute force
for _ in range(400):
    x = Fraction(random.randint(-5000, 5000), random.randint(1, 3000))
    xs = str(float(x)) if random.random() < 0.5 else f"{x.numerator / x.denominator:.4f}"
    v = Fraction(xs); N = random.randint(1, 60)
    best = min((Fraction(round(v * q), q) for q in range(1, N + 1)), key=lambda f: (abs(f - v), f.denominator))
    # also check neighbours of round()
    cands = [Fraction(p, q) for q in range(1, N + 1) for p in (int(v * q) - 1, int(v * q), int(v * q) + 1)]
    best = min(cands, key=lambda f: (abs(f - v), f.denominator))
    assert ns["best_fraction"](xs, N) == f"{best.numerator}/{best.denominator}", (xs, N)
# expand_poly: evaluate at points against Python eval
for e in ["(x+1)^2 - 1", "(x+1)^3", "-(x-1)*(x+1)", "(2*x-3)^2-4*x^2", "(x-1)^10", "--x", "2*-x", "-x^2", "x*x*x - x^3 + 5"]:
    out = ns["expand_poly"](e)
    for xv in range(-3, 4):
        lhs = eval(e.replace("^", "**").replace("x", f"({xv})"))
        rhs_src = re.sub(r"(\d)x", r"\1*x", out).replace("^", "**").replace("x", f"({xv})")
        assert lhs == eval(rhs_src), (e, out, xv)
# count_true: compare with brute force via Python translation using precedence-correct parenthesization
def ct_bf(expr, f):
    vs = sorted(set(re.findall(r"[a-z]", expr)))
    return sum(1 for bits in itertools.product([0, 1], repeat=len(vs)) if f(**dict(zip(vs, bits))))
assert ns["count_true"]("a -> b -> c") == ct_bf("abc", lambda a, b, c: (not a) or ((not b) or c))
assert ns["count_true"]("(a -> b) -> c") == ct_bf("abc", lambda a, b, c: (not ((not a) or b)) or c)
assert ns["count_true"]("a | b ^ c") == ct_bf("abc", lambda a, b, c: a or (b != c))
assert ns["count_true"]("a & b ^ a | b") == ct_bf("ab", lambda a, b: ((a and b) != a) or b)
assert ns["count_true"]("a^b&c") == ct_bf("abc", lambda a, b, c: a != (b and c))
assert ns["count_true"]("x -> y & !x -> z") == ct_bf("xyz", lambda x, y, z: (not x) or ((not (y and not x)) or z))
print("all algo checks passed")
print(ns["prime_josephus"](5), ns["best_fraction"]("3.14159", 100), ns["even_heavy_count"](1, 30), ns["calm_numbers"](100))
