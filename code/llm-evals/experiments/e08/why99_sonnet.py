import json, sys, ast
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify
R = "/home/amit/.cache/vidyalai-scratch/e08/runs/"; D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
he = json.load(open(D + "e08-humaneval.json")); hp = json.load(open(D + "e08-humanevalplus.json"))
recs = {json.loads(l)["id"]: json.loads(l) for l in open(R + "he_sonnet.jsonl")}
for sid in sys.argv[1:]:
    t = int(sid[2:5]); code = verify.extract_code(recs[sid]["text"], he[t]["entry_point"])
    body = hp[t]["test"].split("def check(candidate):")[1]
    inputs = body.split("inputs = ")[1].split("\n")[0]; results = body.split("results = ")[1].split("\n")[0]
    src = verify.build_program(he[t]["prompt"], code) + f"\ninputs = {inputs}\nresults = {results}\nfor inp, exp in zip(inputs, results):\n    got = {he[t]['entry_point']}(*inp)\n    if got != exp:\n        print(repr(inp)[:200], repr(exp), repr(got)); break\nraise SystemExit(1)\n"
    import subprocess, tempfile, pathlib
    with tempfile.TemporaryDirectory(dir=verify.SANDBOX) as d:
        p = pathlib.Path(d) / "prog.py"; p.write_text(verify.GUARD.format(cpu=20) + src)
        res = subprocess.run(verify.UNSHARE + [verify.PYTHON, "-I", str(p)], cwd=d, capture_output=True, text=True, timeout=20, env={"PATH": "/usr/bin:/bin", "HOME": d})
        print(sid, res.stdout[-400:], res.stderr[-300:])
