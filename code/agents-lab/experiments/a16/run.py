"""Part 16 experiment runner: review every corpus page with one configuration, n repeats.

python run.py <config> <repeats> [page ...]

Resumable: a run whose result file exists is skipped; a run that crashed resumes from its checkpoint.
At most 3 requests in flight (3 worker threads). Every call goes through ledger.create (part a16).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import finale  # noqa: E402
import ledger  # noqa: E402
from seeds import CORPUS  # noqa: E402

A16 = Path(__file__).parent
CONFIGS = {
    "haiku-full": dict(model="claude-haiku-5-5", effort="low", kit="full", max_cost=0.05, corpus="seeded"),
    "haiku-full2": dict(model="claude-haiku-5-5", effort="low", kit="full2", max_cost=0.05, corpus="seeded"),
    "haiku-read": dict(model="claude-haiku-5-5", effort="low", kit="read", max_cost=0.05, corpus="seeded"),
    "haiku-full-med": dict(model="claude-haiku-5-5", effort="medium", kit="full", max_cost=0.08, corpus="seeded"),
    "haiku-clean-med": dict(model="claude-haiku-5-5", effort="medium", kit="full", max_cost=0.08, corpus="clean"),
    "haiku-clean": dict(model="claude-haiku-5-5", effort="low", kit="full", max_cost=0.05, corpus="clean"),
    "haiku-hard": dict(model="claude-haiku-5-5", effort="low", kit="full", max_cost=0.05, corpus="hard"),
    "haiku-hard-med": dict(model="claude-haiku-5-5", effort="medium", kit="full", max_cost=0.08, corpus="hard"),
    "sonnet-hard": dict(model="claude-sonnet-5-5", effort="low", kit="full", max_cost=0.60, corpus="hard"),
    "sonnet-clean": dict(model="claude-sonnet-5-5", effort="low", kit="full", max_cost=0.60, corpus="clean"),
    "sonnet-full": dict(model="claude-sonnet-5-5", effort="low", kit="full", max_cost=0.60, corpus="seeded"),
}


def one(cfg_name, page, rep, client):
    cfg = CONFIGS[cfg_name]
    out = A16 / "runs" / cfg_name
    slug = page.replace("/", "__").removesuffix(".rst")
    res_path = out / f"{slug}-r{rep}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    for d in ("ck", "tx"):
        (out / d).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        findings, result, decisions = finale.review_page(
            A16 / "corpus" / cfg["corpus"], page, part="a16", client=client, model=cfg["model"],
            effort=cfg["effort"], kit=cfg["kit"], max_cost=cfg["max_cost"],
            checkpoint=out / "ck" / f"{slug}-r{rep}.json", transcript=out / "tx" / f"{slug}-r{rep}.jsonl")
    except ledger.BudgetExceeded as e:
        print("BUDGET", e, flush=True)
        raise
    wall = time.perf_counter() - t0
    rec = dict(config=cfg_name, page=page, rep=rep, findings=findings, stop=result.stop,
               answer=result.answer, steps=len(result.steps), cost=result.cost, wall_s=round(wall, 2),
               model_s=round(sum(s.latency_s for s in result.steps), 2),
               tokens_in=sum(s.context for s in result.steps), tokens_out=sum(s.output_tokens for s in result.steps),
               last_context=result.steps[-1].context if result.steps else 0,
               tool_calls=sum(len(s.tool_calls) for s in result.steps),
               decisions=decisions)
    res_path.write_text(json.dumps(rec, indent=1))
    print(f"{cfg_name} {page} r{rep}: {len(findings)} findings, {rec['steps']} steps, ${rec['cost']:.4f}, "
          f"{wall:.1f}s, {result.stop}", flush=True)
    return rec


def main():
    cfg_name, reps = sys.argv[1], int(sys.argv[2])
    pages = sys.argv[3:] or CORPUS
    client = ledger.client().with_options(max_retries=6, timeout=180)
    jobs = [(p, r) for r in range(reps) for p in pages]
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda j: one(cfg_name, j[0], j[1], client), jobs))
    print(ledger.report())


if __name__ == "__main__":
    main()
