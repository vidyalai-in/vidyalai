"""Per-run behaviour from transcripts: localization steps, edit errors, test use, repro tests.

Joins runs/results.jsonl with runs/transcripts/<tag>.jsonl and prints per-arm averages.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

A08 = Path("/home/amit/.cache/vidyalai-scratch/a08")
EDIT = {"str_replace", "write_file", "create_file"}


def tag(r):
    return f"{r['task']}-{r['arm']}-{r['model'].split('-')[1]}-{r['effort']}-r{r['rep']}"


def behaviour(r):
    path = A08 / "runs/transcripts" / f"{tag(r)}.jsonl"
    calls, results, step_of = [], {}, {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "step":
            for b in e["content"]:
                if b["type"] == "tool_use":
                    calls.append((e["n"], b["name"], b["input"], b["id"]))
        elif e["event"] == "tool_results":
            for res in e["results"]:
                results[res["tool_use_id"]] = res
    task = json.loads((A08 / "tasks" / r["task"] / "task.json").read_text())
    src_edit_steps = [n for n, name, inp, _ in calls if name in ("str_replace", "write_file")
                      and inp.get("path", "").startswith(task["src_prefix"])]
    edit_errors = sum(1 for n, name, inp, i in calls if name in EDIT and results.get(i, {}).get("is_error"))
    edit_calls = sum(1 for n, name, inp, i in calls if name in EDIT)
    test_steps = [n for n, name, inp, _ in calls if name == "run_tests"]
    last_edit = max(src_edit_steps, default=0)
    tool_chars = sum(len(str(res.get("content", ""))) for res in results.values())
    return dict(
        first_src_edit=min(src_edit_steps, default=None),
        edit_calls=edit_calls, edit_errors=edit_errors,
        test_calls=len(test_steps), tested_after_edit=any(s > last_edit for s in test_steps) if last_edit else False,
        repro=sum(1 for n, name, inp, _ in calls if name == "create_file"),
        touched_gold=any(f in r["files_changed"] for f in task["src_files"]),
        extra_src=[f for f in r["files_changed"] if f not in task["src_files"] and not f.startswith(task["tests_prefix"])],
        tool_chars=tool_chars,
        reads=sum(1 for c in calls if c[1] == "read_file"),
        searches=sum(1 for c in calls if c[1] in ("grep", "glob", "list_dir")),
    )


def main(path=A08 / "runs/results.jsonl"):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    g = defaultdict(list)
    for r in rows:
        r.update(behaviour(r))
        g[(r["model"].split("-")[1], r["arm"])].append(r)
    print(f"{'group':28s} {'n':>4s} {'1stEdit':>7s} {'search':>6s} {'reads':>5s} {'edits':>5s} {'edErr':>5s} "
          f"{'tests':>5s} {'%tAft':>5s} {'%repro':>6s} {'%gold':>5s} {'kchars':>6s}")
    for k in sorted(g):
        rs = g[k]
        fe = [r["first_src_edit"] for r in rs if r["first_src_edit"]]
        print(f"{str(k):28s} {len(rs):4d} {st.mean(fe) if fe else 0:7.1f} {st.mean(r['searches'] for r in rs):6.1f} "
              f"{st.mean(r['reads'] for r in rs):5.1f} {st.mean(r['edit_calls'] for r in rs):5.1f} "
              f"{sum(r['edit_errors'] for r in rs):5d} {st.mean(r['test_calls'] for r in rs):5.1f} "
              f"{100 * st.mean(r['tested_after_edit'] for r in rs):5.0f} {100 * st.mean(r['repro'] > 0 for r in rs):6.0f} "
              f"{100 * st.mean(r['touched_gold'] for r in rs):5.0f} {st.mean(r['tool_chars'] for r in rs) / 1000:6.1f}")
    return rows


if __name__ == "__main__":
    main(*(sys.argv[1:]))
