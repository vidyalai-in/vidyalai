"""Run every HumanEval reply against the original tests and the HumanEval+ tests, in the sandbox.
Usage (inside systemd-run MemoryMax=512M TasksMax=64): grade_code.py <runs/he_X.jsonl> ..."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify
D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
he = json.load(open(D + "e08-humaneval.json")); hp = json.load(open(D + "e08-humanevalplus.json"))
# The Hugging Face copy of HumanEval+ has a broken check for HumanEval/32 (it unpacks the float the
# candidate returns); its canonical solution fails it. Fixed oracle: the returned x must be a root, i.e.
# |poly(xs, x)| <= 1e-4, or no worse than 10x the residual of the reference root (7 of the 886 inputs have
# coefficients near 1e9, where even the reference root leaves a residual above 1e-4).
BAD = "assert _poly(*candidate(*inp), inp) <= 0.0001"
assert BAD in hp[32]["test"]
hp[32] = dict(hp[32], test=hp[32]["test"].replace(BAD, "assert abs(_poly(*inp, candidate(*inp))) <= max(0.0001, 10 * abs(_poly(*inp, exp)))"))

def job(rec):
    i = int(rec["id"][2:5])
    code = verify.extract_code(rec.get("text") or "", he[i]["entry_point"])
    t0 = time.time(); b = verify.humaneval_check(he[i], code, timeout=10)
    t1 = time.time(); p = verify.humaneval_check(hp[i], code, timeout=30)
    return dict(id=rec["id"], task=i, base=b[0], base_err=b[1], plus=p[0], plus_err=p[1],
                t_base=round(t1 - t0, 2), t_plus=round(time.time() - t1, 2),
                stop=rec.get("stop") or ("max_tokens" if rec.get("truncated") else "end_turn"))

for path in sys.argv[1:]:
    recs = [json.loads(l) for l in open(path)]
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(job, recs))
    out = Path(path).with_suffix(".graded.json")
    json.dump(res, open(out, "w"))
    nb = sum(r["base"] == "pass" for r in res); npl = sum(r["plus"] == "pass" for r in res)
    print(path, len(res), "base", nb, "plus", npl, "secs", int(time.time() - t0), flush=True)
