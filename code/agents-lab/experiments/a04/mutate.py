"""Plant one-token bugs in correct solutions Haiku wrote (main run), for the repair experiment.

Mutation operators (applied to one token, formatting preserved):
  comparison swap  <  <=  >  >=  ==  !=   (< <-> <=, > <-> >=, == <-> !=)
  integer constant  n -> n + 1 or n - 1   (n >= 0)
  boolean operator  and <-> or
A mutant qualifies if it still loads and fails at least one, but at most 40%, of the hidden tests.
One mutant per task, chosen at random (seed 4) among qualifying ones.
"""
import io
import json
import random
import sys
import tokenize
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a04")
import reflect as R  # noqa: E402
from sandbox import run_tests  # noqa: E402

SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}


def mutants(code):
    lines = code.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        s = offs[tok.start[0] - 1] + tok.start[1]
        e = offs[tok.end[0] - 1] + tok.end[1]
        reps = []
        if tok.type == tokenize.OP and tok.string in SWAP:
            reps = [SWAP[tok.string]]
        elif tok.type == tokenize.NUMBER and tok.string.isdigit():
            n = int(tok.string)
            reps = [str(n + 1)] + ([str(n - 1)] if n > 0 else [])
        elif tok.type == tokenize.NAME and tok.string in ("and", "or"):
            reps = ["or" if tok.string == "and" else "and"]
        for r in reps:
            out.append(dict(line=tok.start[0], before=tok.string, after=r, code=code[:s] + r + code[e:]))
    return out


def main():
    rows = [json.loads(l) for l in open("/home/amit/.cache/vidyalai-scratch/a04/runs/main.jsonl")]
    T = {t["id"]: t for t in R.TASKS}
    rng = random.Random(4)
    picked = []
    for task in sorted(T):
        src = next((r for r in sorted(rows, key=lambda r: r["rep"]) if r["task"] == task and r["single"]["ok"]), None)
        if src is None:
            continue
        cands = mutants(src["single"]["code"])
        rng.shuffle(cands)
        t = T[task]

        def check(m):
            res = run_tests(m["code"], task, t["tests"], R.EXPECTED[task], timeout=4)
            nfail = sum(not x["ok"] for x in res)
            loads = not any("failed to load" in x["got"] for x in res)
            return m, nfail, loads, all(x["ok"] for x in res[:t["n_public"]])

        ok = []
        for i in range(0, min(len(cands), 64), 8):  # check in batches of 8, stop at the first qualifying one
            with ThreadPoolExecutor(8) as ex:
                results = list(ex.map(check, cands[i:i + 8]))
            ok = [(m, nf, pub) for m, nf, loads, pub in results
                  if loads and 1 <= nf <= 0.4 * len(t["tests"]) and "timed out" not in str(nf)]
            if ok:
                break
        if not ok:
            print(f"{task:22s} no qualifying mutant among {len(cands)}")
            continue
        m, nf, pub = ok[0]
        picked.append(dict(task=task, tag=f"{task}#mut", rep_src=src["rep"], line=m["line"], before=m["before"],
                           after=m["after"], hidden_fails=nf, public_ok=pub, code=m["code"],
                           reply=f"```python\n{m['code']}```"))
        print(f"{task:22s} line {m['line']:3d}: {m['before']!r:>6} -> {m['after']!r:<6} fails {nf:2d}/{len(t['tests'])} "
              f"public {'pass' if pub else 'FAIL'}", flush=True)
    json.dump(picked, open("/home/amit/.cache/vidyalai-scratch/a04/runs/mutants.json", "w"))
    print(len(picked), "mutants;", sum(not p["public_ok"] for p in picked), "caught by public tests")


if __name__ == "__main__":
    main()
