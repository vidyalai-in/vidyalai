"""The <=100K grid for Claude Haiku 5.5, run through the Message Batches API with 1-hour prompt caching.

  grid.py build            build every haystack, count its tokens (free), save runs/grid-haystacks.json
  grid.py submit <stage>   primer | small | large  (writes the batch id to runs/grid-<stage>.batch)
  grid.py collect <stage>  poll until ended, record every result in the ledger, save runs/grid-<stage>.jsonl
"""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import ledger, longctx as lc

RUNS = "/home/amit/.cache/vidyalai-scratch/e09/runs"
MODEL = "claude-haiku-5-5"
LENGTHS = [0, 4000, 8000, 16000, 32000, 64000, 96000]
H, K = 4, 6
MAXTOK = {"literal": 48, "onehop": 48, "vt": 256, "know": 24}


def haystack_specs():
    for li, L in enumerate(LENGTHS):
        for h in range(H):
            depths = [(j * H + h) / (K * H - 1) for j in range(K)]
            yield dict(L=L, h=h, seed=1000 + 10 * li + h, start=(li * H + h) * 389 % 10000, depths=depths)


def build():
    paras = lc.load_corpus()
    cl = ledger.client()
    out = []
    for s in haystack_specs():
        n = int((s["L"] - 1050) * 3.07) if s["L"] else 0
        while True:
            hay = lc.build_haystack(paras, n, s["start"], s["seed"], k=K, char_depths=s["depths"])
            q = lc.questions(hay)[0][2]
            ntok = cl.messages.count_tokens(**{k: v for k, v in lc.claude_params(hay["text"], q).items()
                                               if k in ("model", "messages", "thinking")}).input_tokens
            if s["L"] == 0 or ntok <= min(s["L"] * 1.03, 99_000):
                break
            n = int(n * 0.98)
        hay.update(s, n_tokens=ntok)
        out.append(hay)
        print(s["L"], s["h"], len(hay["text"]), ntok, flush=True)
    json.dump(out, open(f"{RUNS}/grid-haystacks.json", "w"))


def requests(stage):
    hays = json.load(open(f"{RUNS}/grid-haystacks.json"))
    reqs = []
    for hay in hays:
        qs = lc.questions(hay)
        if stage == "primer":
            if hay["L"] == 0:
                pick = qs  # tiny, uncached: run all of them now
            else:
                pick = qs[:1]
        elif stage == "small":
            pick = qs[1:] if 0 < hay["L"] <= 32000 else []
        elif stage == "large":
            pick = qs[1:] if hay["L"] > 32000 else []
        elif stage in ("sprimer", "srest"):  # Sonnet 5.5 headline comparison at 8K, all tasks
            sel = qs if hay["L"] == 8000 else []
            pick = sel[:1] if stage == "sprimer" else sel[1:]
        elif stage in ("tprimer", "trest"):  # thinking arm: one-hop and VT only, three lengths
            sel = [q for q in qs if q[1] in ("onehop", "vt")] if hay["L"] in (8000, 32000, 96000) else []
            pick = sel[:1] if stage == "tprimer" else sel[1:]
        for qid, task, q, gold, depth in pick:
            p = lc.claude_params(hay["text"], q, max_tokens=MAXTOK[task], cache=hay["L"] > 0, ttl="1h")
            if stage.startswith("s"):  # Sonnet 5.5 rejects "disabled"; "between_tools" turns off up-front thinking
                p.update(model="claude-sonnet-5-5", thinking={"type": "between_tools"})
            if stage.startswith("t"):  # adaptive thinking at the default effort (medium); room to think
                p.update(thinking={"type": "adaptive"}, max_tokens=4000)
            reqs.append((dict(custom_id=f"L{hay['L']}-h{hay['h']}-{qid}", params=p),
                         dict(L=hay["L"], h=hay["h"], qid=qid, task=task, gold=gold, depth=depth, n_tokens=hay["n_tokens"])))
    if stage == "primer":  # world-knowledge control: does the model know each landmark's city?
        for i, (lm, city) in enumerate(lc.LANDMARKS):
            p = dict(model=MODEL, max_tokens=MAXTOK["know"], thinking={"type": "disabled"},
                     messages=[{"role": "user", "content": lc.Q_KNOW.format(landmark=lm)}])
            reqs.append((dict(custom_id=f"know-{i}", params=p), dict(L=-1, h=-1, qid=f"know-{i}", task="know", gold=city,
                                                                    depth=None, n_tokens=30)))
    return reqs


def estimate(reqs):
    """Worst case at batch price: every prompt token written to a 1-hour cache (2x input) and
    max_tokens of output. Never below the ledger's own estimate."""
    tot = 0.0
    for r, m in reqs:
        n = m["n_tokens"] + 120
        model = r["params"]["model"]
        rates = ledger._rates(model, n)
        mine = (n * 2 * rates["in"] + r["params"]["max_tokens"] * rates["out"]) / 1e6
        tot += max(mine, ledger.estimate_cost(model, n, r["params"]["max_tokens"]))
    return tot * 0.5


def submit(stage):
    reqs = requests(stage)
    est = estimate(reqs)
    print(stage, len(reqs), "requests, worst case $%.4f" % est)
    ledger.guard("e09", est)
    cl = ledger.client()
    b = cl.messages.batches.create(requests=[r for r, _ in reqs])
    json.dump(dict(id=b.id, stage=stage, n=len(reqs), est=est, t0=time.time()), open(f"{RUNS}/grid-{stage}.batch", "w"))
    json.dump({r["custom_id"]: m for r, m in reqs}, open(f"{RUNS}/grid-{stage}.meta.json", "w"))
    print("batch", b.id)


def collect(stage):
    info = json.load(open(f"{RUNS}/grid-{stage}.batch"))
    meta = json.load(open(f"{RUNS}/grid-{stage}.meta.json"))
    cl = ledger.client()
    while True:
        b = cl.messages.batches.retrieve(info["id"])
        if b.processing_status == "ended":
            break
        print(time.strftime("%H:%M:%S"), b.request_counts.model_dump(), flush=True)
        time.sleep(20)
    secs = time.time() - info["t0"]
    n_ok = cost = 0
    with open(f"{RUNS}/grid-{stage}.jsonl", "w") as f:
        for item in cl.messages.batches.results(info["id"]):
            m = meta[item.custom_id]
            rec = dict(custom_id=item.custom_id, **m, result=item.result.type, batch=True)
            if item.result.type == "succeeded":
                msg = item.result.message
                cost += ledger.record("e09", msg.model if msg.model in ledger.ALLOWED_MODELS else
                                      ("claude-sonnet-5-5" if stage.startswith("s") else MODEL), msg.usage,
                                      note=f"grid {stage}", batch=True)
                n_ok += 1
                rec.update(reply="".join(c.text for c in msg.content if c.type == "text"), stop=msg.stop_reason,
                           thinking="".join(getattr(c, "thinking", "") or "" for c in msg.content if c.type == "thinking"),
                           usage=msg.usage.model_dump())
            else:
                rec.update(error=item.result.model_dump(exclude_none=True))
            f.write(json.dumps(rec) + "\n")
    print(f"{stage}: {n_ok}/{len(meta)} succeeded, ${cost:.4f}, {secs:.0f} s after submit")


if __name__ == "__main__":
    {"build": lambda: build(), "submit": lambda: submit(sys.argv[2]), "collect": lambda: collect(sys.argv[2])}[sys.argv[1]]()
