"""Part 1 experiments. Arms:
  agent        harness loop, Haiku 5.5, caching on       (all tasks x R repeats)
  agent_nocache  same, caching off                      (subset)
  rag          workflow: retrieve top-3 files by TF-IDF, then one call   (all tasks x R)
  stuff        workflow: whole folder in the prompt, one call (cached)   (all tasks x R)
  sonnet       harness loop, Sonnet 5.5, caching on      (subset)
Usage: python run_experiments.py ARM REPEATS [task ids...]
Writes runs/<arm>.jsonl (one summary line per run) and runs/transcripts/<arm>/<task>-r<k>.jsonl.
"""
import json, math, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
from harness import _TimedClient  # noqa: E402
from docs_agent import ROOT, SYSTEM, TOOLS, grade, run_agent  # noqa: E402

HERE = Path(__file__).parent
TASKS = json.loads((HERE / "tasks.json").read_text())
FILES = sorted(f for f in ROOT.rglob("*") if f.is_file())
DOCS = {str(f.relative_to(ROOT)): f.read_text() for f in FILES}
CL = ledger.client()

WF_SYSTEM = """You answer questions about Larkspur Robotics, a fictional company, using excerpts from its internal docs folder.
Use only the provided files. Do not guess and do not rely on outside knowledge.
For counts and totals over a CSV file, go through every row carefully.
Reply with one or two sentences of explanation, then a last line in exactly this form:
ANSWER: <the answer>
If the files do not contain the answer, write ANSWER: unknown"""

STOP = set("a an the of in on for to and or is are was what which who how many much does do did that with by at "
           "from it its this be has have when whose there their than after before".split())

def toks(s):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-\.#]*", s.lower()) if w not in STOP]

DF = Counter(w for text in DOCS.values() for w in set(toks(text)))

def retrieve(q, k=3):
    """Plain TF-IDF over whole files (path included): the fixed 'retrieve' step of the workflow."""
    qs = set(toks(q))
    def score(path, text):
        tf = Counter(toks(path + " " + text))
        return sum((1 + math.log(tf[w])) * math.log(len(DOCS) / DF.get(w, 1)) for w in qs if tf[w])
    ranked = sorted(DOCS, key=lambda p: -score(p, DOCS[p]))
    return ranked[:k]

def one_call(part, model, system_blocks, user, note):
    cl = _TimedClient(CL)
    r = ledger.create(cl, part, note=note, model=model, max_tokens=8000, system=system_blocks,
                      messages=[{"role": "user", "content": user}], thinking={"type": "adaptive"},
                      output_config={"effort": "medium"})
    text = "".join(b.text for b in r.content if b.type == "text")
    u = r.usage
    return text, {"input_tokens": u.input_tokens, "cache_read": u.cache_read_input_tokens or 0,
                  "cache_write": u.cache_creation_input_tokens or 0, "output_tokens": u.output_tokens,
                  "latency_s": round(cl.last, 3), "cost": ledger.cost_of(model, u)}

def files_block(paths):
    return "\n\n".join(f'<file path="{p}">\n{DOCS[p]}</file>' for p in paths)

def run_one(arm, task, k):
    tdir = HERE / "runs" / "transcripts" / arm
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{task['id']}-r{k}.jsonl"
    tpath.unlink(missing_ok=True)
    t0 = time.perf_counter()
    if arm.startswith("agent") or arm == "sonnet":
        model = "claude-sonnet-5-5" if arm == "sonnet" else "claude-haiku-5-5"
        r = run_agent(task["question"], TOOLS, SYSTEM, part="a01", model=model, effort="medium",
                      cache=("nocache" not in arm), parallel=("serial" not in arm),
                      max_steps=(30 if "serial" in arm else 12), transcript=tpath, verbose=False)
        rec = {"answer": r.answer, "stop": r.stop, "cost": r.cost, "steps": [s.__dict__ for s in r.steps]}
    else:
        if arm == "rag":
            got = retrieve(task["question"])
            sysb = [{"type": "text", "text": WF_SYSTEM}]
            user = files_block(got) + "\n\nQuestion: " + task["question"]
        else:  # stuff: the whole folder, cached as part of the system prompt
            got = list(DOCS)
            sysb = [{"type": "text", "text": WF_SYSTEM + "\n\nThe docs folder:\n\n" + files_block(got),
                     "cache_control": {"type": "ephemeral"}}]
            user = "Question: " + task["question"]
        text, st = one_call("a01", "claude-haiku-5-5", sysb, user, f"{arm} {task['id']}")
        rec = {"answer": text, "stop": "end_turn", "cost": st["cost"], "steps": [st], "retrieved": got}
        with tpath.open("w") as f:
            f.write(json.dumps({"event": "workflow", "arm": arm, "task": task, "retrieved": got,
                                "system": sysb[0]["text"][:400], "user": user[-300:], "answer": text, **st}) + "\n")
    rec.update(arm=arm, task=task["id"], kind=task["kind"], repeat=k, correct=grade(task, rec["answer"]),
               wall_s=round(time.perf_counter() - t0, 2))
    return rec

if __name__ == "__main__":
    arm, R = sys.argv[1], int(sys.argv[2])
    ids = set(sys.argv[3:])
    tasks = [t for t in TASKS if not ids or t["id"] in ids]
    jobs = [(t, k) for k in range(1, R + 1) for t in tasks]
    out = HERE / "runs" / f"{arm}.jsonl"
    with ThreadPoolExecutor(max_workers=4) as ex:
        for rec in ex.map(lambda j: run_one(arm, *j), jobs):
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{arm} {rec['task']} r{rec['repeat']} correct={rec['correct']} steps={len(rec['steps'])} "
                  f"${rec['cost']:.5f} stop={rec['stop']}", flush=True)
    print(ledger.report())
