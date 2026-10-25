"""Run arms x questions x repeats, resumable, at most 3 requests in flight.
usage: run.py ARM MODEL REPEATS [question ids...]"""
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
import research  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"
Q = {q["id"]: q for q in json.loads((HERE / "questions.json").read_text())}
lock = threading.Lock()


def grade(q, answer):
    """Numbers, dates and versions must open the answer; names and lists may appear anywhere."""
    a = re.sub(r"^(JEP|about|approximately)\s+", "", answer.strip().rstrip(".").lstrip("~≈ "), flags=re.I)
    if re.match(r"(unable|cannot|can't|unknown|not (yet )?(determin|known|found))", a, re.I):
        return False
    for p in q["accept"]:
        if p[0].isdigit() or p[0] in "-−":
            if re.match(p + r"(?![\d.])", a, re.I):
                return True
        elif re.search(p, a, re.I):
            return True
    return False


def done():
    if not RESULTS.exists():
        return set()
    return {(r["arm"], r["model"], r["qid"], r["rep"]) for r in map(json.loads, RESULTS.read_text().splitlines())}


def one(arm, model, qid, rep, client):
    q = Q[qid]
    tpath = HERE / "runs" / f"{arm}-{model}.jsonl"
    try:
        out = research.research(q["q"], arm, model=model, transcript=tpath, client=client)
    except ledger.BudgetExceeded as e:
        print("BUDGET", e)
        return
    except Exception as e:  # log and move on; the run can be resumed
        print("ERROR", arm, qid, rep, type(e).__name__, e)
        return
    rec = dict(arm=arm, model=model, qid=qid, kind=q["kind"], rep=rep, correct=grade(q, out["answer"]),
               **{k: v for k, v in out.items() if k != "text"})
    with lock:
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"{arm:12s} {model:18s} {qid} r{rep} {'OK ' if rec['correct'] else 'BAD'} "
          f"s={len(out['searches'])} f={len(out['fetches'])} ${out['cost']:.4f} {out['latency_s']}s  {out['answer'][:60]}")


if __name__ == "__main__":
    arm, model, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ids = sys.argv[4:] or list(Q)
    have = done()
    jobs = [(arm, model, qid, r) for r in range(reps) for qid in ids if (arm, model, qid, r) not in have]
    client = ledger.client().with_options(max_retries=6)
    with ThreadPoolExecutor(3) as ex:
        for j in jobs:
            ex.submit(one, *j, client)
    print(ledger.report())
