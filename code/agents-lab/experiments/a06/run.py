"""Tool-choice accuracy as the number of available tools grows.

Arms (all single-turn: one request in, the first real tool call out):
  all          every one of the N tools loaded in the prompt
  search_bm25  the API's tool search tool (BM25 variant), all N tools sent with defer_loading
  search_regex the regex variant of the same
  client_bm25  our own BM25 over the N tools picks the top K, only those are loaded

usage: python run.py ARM N [--model M] [--reps R] [--only id,id] [--effort E] [--tag T]
Results append to runs/<tag>.jsonl; finished (arm, N, model, effort, id, rep) rows are skipped.
"""

import argparse
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger  # noqa: E402
from catalog import CATALOG, BY_NAME, api_tool  # noqa: E402
from requests_set import REQUESTS, SYSTEM, check  # noqa: E402

HERE = Path(__file__).parent
SEARCH_TOOLS = {"search_bm25": {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
                "search_regex": {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}}


def toolset(r, n):
    """The target plus n-1 others, chosen by a per-request seeded shuffle (so sets are nested:
    the 20-tool set contains the 5-tool set), kept in catalog order (server by server)."""
    others = [t["name"] for t in CATALOG if t["name"] != r["tool"]]
    random.Random(r["id"]).shuffle(others)
    keep = set(others[:n - 1]) | {r["tool"]}
    return [t for t in CATALOG if t["name"] in keep]


# ---- client-side BM25 over tool text ----------------------------------------------------
STOP = set("a an the of to in on for and or by with from as at is be it this that its all any one".split())


def tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower().replace("_", " ")) if w not in STOP]


def tool_text(t):
    props = t["input_schema"]["properties"]
    return " ".join([t["name"], t["description"]] + [f"{k} {v.get('description', '')}" for k, v in props.items()])


def bm25_rank(query, tools, k1=1.2, b=0.75):
    docs = [tokens(tool_text(t)) for t in tools]
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    scores = []
    for t, d in zip(tools, docs):
        tf = Counter(d)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
                / (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in set(tokens(query)) if w in tf)
        scores.append(s)
    return [t for _, t in sorted(zip(scores, tools), key=lambda x: -x[0])]


# ---- one trial --------------------------------------------------------------------------
WIDE = SYSTEM.replace("Fulfil each request", "Other connected servers belong to other teams and acquired companies; use them only when a request names them.\nFulfil each request")
HINT = ("Tool search returns at most 5 tools by default, in catalog order, so a broad pattern can miss the tool you need. "
        "If the results don't include a tool that does exactly what was asked, search again with a narrower pattern "
        "(for example the action word plus the service name) or a higher limit before concluding no such tool exists.")
FIXED = {}  # set by --catalog: one fixed tool list (base + generated servers) for every request


def build(arm, r, n, k):
    tools = FIXED.get("tools") or toolset(r, n)
    system = [{"type": "text", "text": FIXED.get("system", SYSTEM)}]
    if arm == "all":
        api_tools = [api_tool(t) for t in tools]
        if FIXED or n == len(CATALOG):  # the same full catalog every time: cache tools + system
            system[0]["cache_control"] = {"type": "ephemeral"}
    elif arm in SEARCH_TOOLS:
        api_tools = [SEARCH_TOOLS[arm]] + [api_tool(t, defer=True) for t in tools]
        system[0]["cache_control"] = {"type": "ephemeral"}  # search tool + system are a stable prefix
    elif arm == "client_bm25":
        api_tools = [api_tool(t) for t in bm25_rank(r["text"], tools)[:k]]
    else:
        raise ValueError(arm)
    return api_tools, system


def trial(cl, arm, r, n, k, model, effort, rep, tag):
    api_tools, system = build(arm, r, n, k)
    req = dict(model=model, max_tokens=4000, system=system, tools=api_tools,
               output_config={"effort": effort}, thinking={"type": "adaptive", "display": "summarized"})
    messages = [{"role": "user", "content": r["text"]}]
    usage = Counter()
    content, t0 = [], time.perf_counter()
    for _ in range(3):  # pause_turn: a server-side tool loop paused, send it back to continue
        resp = ledger.create(cl, "a06", note=f"a06 {tag} {arm} n={n}", messages=messages, **req)
        u = resp.usage.model_dump()
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            usage[key] += u.get(key) or 0
        usage["thinking_tokens"] += ((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        usage["cost"] += ledger.cost_of(model, resp.usage)
        content += [b.model_dump(exclude_none=True) for b in resp.content]
        if resp.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": resp.content}]
    latency = time.perf_counter() - t0
    calls = [b for b in content if b["type"] == "tool_use"]
    searches = [b for b in content if b["type"] == "server_tool_use"]
    found = [ref["tool_name"] for b in content if b["type"] == "tool_search_tool_result"
             for ref in (b.get("content") or {}).get("tool_references", [])]
    first = calls[0] if calls else None
    right_tool, right_args = check(r, first["name"], first["input"]) if first else (False, False)
    return {"tag": tag, "arm": arm, "n": n, "k": k, "model": model, "effort": effort, "id": r["id"], "rep": rep,
            "target": r["tool"], "called": first["name"] if first else None, "input": first["input"] if first else None,
            "n_calls": len(calls), "right_tool": right_tool, "right_args": right_args,
            "searches": [b["input"] for b in searches], "found": found, "target_found": r["tool"] in found,
            "loaded": [t["name"] for t in api_tools if not t.get("defer_loading") and "type" not in t],
            "stop": resp.stop_reason, "latency": round(latency, 3), "usage": dict(usage), "content": content}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("n", type=int)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hint", action="store_true", help="tell the model that searches return only the top 5")
    ap.add_argument("--catalog", type=int, default=0, help="145, 300 or 486: fixed catalog incl. generated servers")
    a = ap.parse_args()
    if a.hint:
        FIXED["system"] = SYSTEM + "\n" + HINT
    if a.catalog:
        from gen_catalog import CATALOGS
        FIXED.update(tools=CATALOGS[a.catalog], system=WIDE)
        assert a.n == a.catalog
    out = HERE / "runs" / f"{a.tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["arm"], d["n"], d["model"], d["effort"], d["id"], d["rep"]))
    reqs = [r for r in REQUESTS if not a.only or r["id"] in a.only.split(",")]
    jobs = [(r, rep) for rep in range(a.reps) for r in reqs
            if (a.arm, a.n, a.model, a.effort, r["id"], rep) not in done]
    print(f"{len(jobs)} trials to run -> {out}")
    cl = ledger.client().with_options(max_retries=6)
    lock = threading.Lock()

    def go(job):
        r, rep = job
        try:
            res = trial(cl, a.arm, r, a.n, a.k, a.model, a.effort, rep, a.tag)
        except ledger.BudgetExceeded as e:
            print("BUDGET:", e)
            raise
        except Exception as e:  # log and move on; the row is retried on the next run
            print("ERROR", r["id"], type(e).__name__, str(e)[:300])
            return None
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(res, default=str) + "\n")
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        results = [x for x in ex.map(go, jobs) if x]
    if results:
        ok = sum(x["right_tool"] for x in results)
        okk = sum(x["right_args"] for x in results)
        cost = sum(x["usage"]["cost"] for x in results)
        tok = sum(x["usage"]["input_tokens"] + x["usage"]["cache_read_input_tokens"]
                  + x["usage"]["cache_creation_input_tokens"] for x in results) / len(results)
        print(f"{a.arm} n={a.n}: tool {ok}/{len(results)}  tool+args {okk}/{len(results)}  "
              f"prompt tokens/trial {tok:,.0f}  cost ${cost:.4f}  (${cost / len(results):.5f}/trial)")


if __name__ == "__main__":
    main()
