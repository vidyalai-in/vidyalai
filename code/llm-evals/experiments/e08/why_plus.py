"""For each sample that passes the original tests but fails HumanEval+, find the first failing input (sandboxed)."""
import json, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import verify
R = "/home/amit/.cache/vidyalai-scratch/e08/runs/"; D = "/home/amit/.cache/vidyalai-scratch/evals/data/"
he = json.load(open(D + "e08-humaneval.json")); hp = json.load(open(D + "e08-humanevalplus.json"))
name = sys.argv[1]
g = json.load(open(R + f"he_{name}.graded.json")); recs = {json.loads(l)["id"]: json.loads(l) for l in open(R + f"he_{name}.jsonl")}
seen = set(); out = []
for r in g:
    t = r["task"]
    if r["base"] != "pass" or r["plus"] == "pass" or t == 32 or (t in seen and name == "haiku"):
        continue
    seen.add(t)
    code = verify.extract_code(recs[r["id"]]["text"], he[t]["entry_point"])
    test = hp[t]["test"].replace("    for i, (inp, exp) in enumerate(zip(inputs, results)):\n",
        "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n        import copy, sys as _s\n        try:\n            _got = candidate(*copy.deepcopy(inp))\n        except Exception as e:\n            _got = repr(e)\n        if True:\n            try:\n                assertion(_got, exp, 1e-6) if 'assertion' in globals() else None\n                ok = True\n            except Exception:\n                ok = False\n            if not ok:\n                print('INPUT', repr(inp)[:300]); print('EXPECTED', repr(exp)[:200]); print('GOT', repr(_got)[:200]); _s.stdout.flush(); raise SystemExit(3)\n        continue\n")
    src = verify.build_program(he[t]["prompt"], code) + "\n" + test + f"\ncheck({he[t]['entry_point']})\n"
    import subprocess, tempfile, pathlib
    with tempfile.TemporaryDirectory(dir=verify.SANDBOX) as d:
        p = pathlib.Path(d) / "prog.py"; p.write_text(verify.GUARD.format(cpu=31) + src)
        try:
            res = subprocess.run(verify.UNSHARE + [verify.PYTHON, "-I", str(p)], cwd=d, capture_output=True, text=True, timeout=30,
                                 env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "HOME": d})
            info = res.stdout[-700:] + res.stderr[-300:]
        except subprocess.TimeoutExpired:
            info = "TIMEOUT"
    out.append(dict(task=t, id=r["id"], code=code, info=info))
    print("=====", he[t]["task_id"], r["id"]); print(code.strip()[-900:]); print("--"); print(info)
json.dump(out, open(R + f"why_plus_{name}.json", "w"), indent=1)
