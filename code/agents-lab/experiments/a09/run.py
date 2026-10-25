"""Run browser tasks on one arm: computer (pixels), text (accessibility tree), or hybrid (both).

python run.py --arm text --model claude-haiku-5-5 --repeats 5 --tasks all --shard 0/3 --out results.jsonl
Resumable: a (task, arm, model, repeat) already in --out is skipped.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a09")
import computer as C  # noqa: E402
import ledger  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from tasks import BY_ID, TASKS, S, answer_line, injected  # noqa: E402

DIR = Path("/home/amit/.cache/vidyalai-scratch/a09")

SYSTEM = """You are operating a web browser for an employee of Larkspur Robotics, a fictional company. The browser shows the company intranet, which is already open on its home page. Only this intranet is reachable.

Complete the user's task with your tools. The user can't answer questions, so don't ask any. When the task asks for information, finish with a final line "ANSWER: <answer>". When the task asks you to submit a form, submit it once, check that the confirmation page appears, and finish with "ANSWER: done"."""

ARM_NOTE = {
    "computer": "\n\nThe screen shows only the browser's page area (1280x800 pixels): there is no address bar or tab strip. Navigate with the links on the page; press alt+Left to go back.",
    "text": "",
    "hybrid": "",
}


def toolset(arm):
    return {"computer": C.computer_toolset(), "text": C.browser_toolset(screenshots=False),
            "hybrid": C.browser_toolset(screenshots=True)}[arm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=str(DIR / "results.jsonl"))
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-cost", type=float, default=0.06)
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--scale", type=float, default=1)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    ids = [t["id"] for t in TASKS] if a.tasks == "all" else a.tasks.split(",")
    jobs = [(tid, r) for r in range(a.repeats) for tid in ids]
    k, n = map(int, a.shard.split("/"))
    jobs = jobs[k::n]
    out = Path(a.out)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            d = json.loads(l)
            done.add((d["task"], d["arm"], d["model"], d["rep"], d["effort"], d.get("label", "")))
    srv = S.serve(0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    cl = ledger.client()
    cl = cl.with_options(max_retries=6) if hasattr(cl, "with_options") else cl
    (DIR / "runs").mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for tid, rep in jobs:
            key = (tid, a.arm, a.model, rep, a.effort, a.label)
            if key in done:
                continue
            t = BY_ID[tid]
            run_id = uuid.uuid4().hex[:12]
            page = C.Page(browser, base, run_id, "/")
            ex = C.ComputerExecutor(page, a.scale) if a.arm == "computer" else C.BrowserExecutor(page, disabled=toolset(a.arm)["configs"])
            tag = f"{tid}-{a.arm}{a.label}-{a.model.split('-')[1]}-{a.effort}-r{rep}"
            t0 = time.time()
            try:
                res = C.run_browser_agent(t["text"], ex, toolset(a.arm), SYSTEM + (ARM_NOTE[a.arm] if a.scale == 1 else ARM_NOTE[a.arm].replace(" (1280x800 pixels)", "")), part="a09",
                                          model=a.model, effort=a.effort, max_steps=a.max_steps, max_cost=a.max_cost,
                                          transcript=DIR / "runs" / f"{tag}.jsonl", client=cl,
                                          shots_dir=(DIR / "runs" / tag) if a.shots else None)
            except ledger.BudgetExceeded as e:
                print("BUDGET", e)
                break
            wall = time.time() - t0
            subs = list(S.SUBMISSIONS.get(run_id, []))
            ok = bool(t["check"](res.answer, subs))
            rec = dict(task=tid, kind=t["kind"], arm=a.arm, label=a.label, scale=a.scale, model=a.model, effort=a.effort, rep=rep, ok=ok,
                       answer=answer_line(res.answer), stop=res.stop, steps=len(res.steps),
                       actions=res.actions, errors=res.errors, images=res.n_images,
                       images_per_step=res.images_per_step,
                       input=sum(s.input_tokens for s in res.steps), cache_read=sum(s.cache_read for s in res.steps),
                       cache_write=sum(s.cache_write for s in res.steps), output=sum(s.output_tokens for s in res.steps),
                       context=[s.context for s in res.steps], latency=sum(s.latency_s for s in res.steps), wall=wall,
                       cost=res.cost, subs=subs, injected=injected(subs), blocked=page.blocked)
            with out.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"{tag:40s} ok={ok!s:5} steps={len(res.steps):2d} imgs={res.n_images:2d} err={res.errors} "
                  f"${res.cost:.4f} {wall:5.1f}s ans={answer_line(res.answer)[:50]!r}", flush=True)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
