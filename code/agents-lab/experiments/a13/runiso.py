import json, os, shutil, sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import sandbox as sb
token, path = sb.make_canary()
os.environ["FAKEAPP_TOKEN"] = token          # the host process holds the fake secret
work = sb.new_workdir("iso")
shutil.copy(sb.SCRATCH / "isocheck.py", work / "isocheck.py")
argv = ["python3", "isocheck.py", token, str(path)]
out = {}
r = sb.run_bwrap(work, argv); out["bwrap"] = json.loads(r.stdout) if r.returncode == 0 else r.stderr[:300]
r = sb.run_docker_fresh(work, argv, f"{sb.PREFIX}-iso"); out["docker"] = json.loads(r.stdout) if r.returncode == 0 else r.stderr[:300]
json.dump(out, open(sb.SCRATCH / "runs" / "isocheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
