"""Summarize every v04 run into results.json (small; committed with the code)."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import vlm, ledger

HERE = os.path.dirname(os.path.abspath(__file__))
items = {i.id: i for i in vlm.make_items(150)}


def load(name):
    """Rows of one run, re-scored from the reply text with the current vlm.score."""
    p = f"{HERE}/{name}"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    for r in rows:
        if "text" in r:
            r.update(vlm.score(items[r["id"]], r["text"]))
    return rows


runs = {
    "haiku": [dict(r, setting="thinking off") for r in load("claude_main.jsonl") if "error" not in r],
    "haiku-think": [dict(r, setting="thinking on") for r in load("claude_haiku_think.jsonl") if "error" not in r],
    "haiku-terse": [dict(r, setting="terse") for r in load("claude_terse.jsonl") if "error" not in r],
    "sonnet": [dict(r, setting="between_tools") for r in load("claude_sonnet.jsonl") if "error" not in r],
    "qwen2vl-2b": load("local_qwen.jsonl"),
    "smolvlm-256m": load("local_smol.jsonl"),
}

out = {"accuracy": {}, "read_by_height": {}, "absent": {}, "dense_within10": {}, "chart_exact": {}, "fail_examples": {},
       "paired": {}, "output_tokens": {}, "declined_read": {}}


def rate(xs):
    k, n = sum(xs), len(xs)
    p, lo, hi = vlm.wilson(k, n)
    return dict(k=k, n=n, p=round(p, 4), lo=round(lo, 4), hi=round(hi, 4))


for model, rows in runs.items():
    g = defaultdict(list)
    for r in rows:
        g[(r["setting"], r["task"], r["size"])].append(r)
    for (setting, task, size), rs in sorted(g.items()):
        key = f"{model}|{setting}"
        out["accuracy"].setdefault(key, {}).setdefault(task, {})[str(size)] = rate([r["ok"] for r in rs])
        if task == "read":
            out["declined_read"].setdefault(key, {})[str(size)] = sum(r["pred"] is None for r in rs)
            for r in rs:
                h = items[r["id"]].meta["font_px"] * size / vlm.MASTER
                out["read_by_height"].setdefault(key, {}).setdefault(f"{h:g}", []).append(r["ok"])
        if task == "absent":
            no = [r for r in rs if r["answer"] == "no"]
            yes = [r for r in rs if r["answer"] == "yes"]
            out["absent"].setdefault(key, {})[str(size)] = dict(
                false_yes=rate([r["pred"] == "yes" for r in no]), miss=rate([r["pred"] != "yes" for r in yes]),
                yes_rate=round(sum(r["pred"] == "yes" for r in rs) / len(rs), 4))
        if task == "dense":
            out["dense_within10"].setdefault(key, {})[str(size)] = rate([bool(r.get("within10")) for r in rs])
            errs = [float(r["pred"]) - float(r["answer"]) for r in rs if r["pred"] is not None]
            out["dense_within10"][key][str(size)]["mean_signed_error"] = round(sum(errs) / max(1, len(errs)), 2)
        if task == "chart":
            out["chart_exact"].setdefault(key, {})[str(size)] = rate([r["exact"] for r in rs])
        if "usage" in rs[0]:
            ot = sorted(r["usage"]["output_tokens"] for r in rs)
            out["output_tokens"].setdefault(key, {}).setdefault(task, {})[str(size)] = dict(
                mean=round(sum(ot) / len(ot), 1), median=ot[len(ot) // 2], max=ot[-1],
                max_tokens_stops=sum(r.get("stop") == "max_tokens" for r in rs))
        bad = [r for r in rs if not r["ok"]][:4]
        out["fail_examples"].setdefault(key, {}).setdefault(task, {})[str(size)] = [
            dict(id=r["id"], answer=r["answer"], text=(r.get("text") or "")[:120]) for r in bad]

for key, d in out["read_by_height"].items():
    out["read_by_height"][key] = {h: rate(v) for h, v in sorted(d.items(), key=lambda kv: float(kv[0]))}


def paired(model_a, set_a, size_a, model_b, set_b, size_b, task):
    A = {r["id"]: r["ok"] for r in runs[model_a] if r["setting"] == set_a and r["size"] == size_a and r["task"] == task}
    B = {r["id"]: r["ok"] for r in runs[model_b] if r["setting"] == set_b and r["size"] == size_b and r["task"] == task}
    ids = sorted(set(A) & set(B))
    if not ids:
        return None
    d, lo, hi = vlm.paired_diff([A[i] for i in ids], [B[i] for i in ids])
    return dict(n=len(ids), diff=round(d, 4), lo=round(lo, 4), hi=round(hi, 4))


for task in vlm.TASKS:
    out["paired"][f"haiku 448->896 {task}"] = paired("haiku", "thinking off", 448, "haiku", "thinking off", 896, task)
    out["paired"][f"haiku 224->448 {task}"] = paired("haiku", "thinking off", 224, "haiku", "thinking off", 448, task)
    out["paired"][f"qwen 224->896 {task}"] = paired("qwen2vl-2b", "default", 224, "qwen2vl-2b", "default", 896, task)
    out["paired"][f"smol 224->896 {task}"] = paired("smolvlm-256m", "default", 224, "smolvlm-256m", "default", 896, task)
    out["paired"][f"smol le512->default @896 {task}"] = paired("smolvlm-256m", "le512", 896, "smolvlm-256m", "default", 896, task)
    out["paired"][f"qwen->haiku @896 {task}"] = paired("qwen2vl-2b", "default", 896, "haiku", "thinking off", 896, task)
    out["paired"][f"haiku->sonnet @448 {task}"] = paired("haiku", "thinking off", 448, "sonnet", "between_tools", 448, task)
    out["paired"][f"haiku->terse @896 {task}"] = paired("haiku", "thinking off", 896, "haiku-terse", "terse", 896, task)
    out["paired"][f"haiku->terse @448 {task}"] = paired("haiku", "thinking off", 448, "haiku-terse", "terse", 448, task)
    out["paired"][f"haiku off->on @224 {task}"] = paired("haiku", "thinking off", 224, "haiku-think", "thinking on", 224, task)
    out["paired"][f"haiku->sonnet @224 {task}"] = paired("haiku", "thinking off", 224, "sonnet", "between_tools", 224, task)
    out["paired"][f"haiku off->on @896 {task}"] = paired("haiku", "thinking off", 896, "haiku-think", "thinking on", 896, task)
out["paired"] = {k: v for k, v in out["paired"].items() if v}

# Claude image tokens: billed usage minus the free count of the same prompt's text alone.
ct = {(r["task"], r["size"]): r for r in json.load(open(f"{HERE}/claude_count_tokens.json"))}
tt = json.load(open(f"{HERE}/text_tokens.json"))
TERSE = " Do not list, count aloud or explain: reply with the number immediately."
tok = {}
for model in ("haiku", "sonnet", "haiku-think", "haiku-terse"):
    mid = "claude-sonnet-5-5" if model == "sonnet" else "claude-haiku-5-5"
    by = defaultdict(list)
    for r in runs[model]:
        it = items[r["id"]]
        q = it.question + " " + vlm.SUFFIX[it.task] + (TERSE if model == "haiku-terse" else "")
        # count_tokens reports 2 more than is billed for the same text-only request (textonly_check.json)
        by[r["size"]].append(r["usage"]["input_tokens"] - (tt[f"{mid}|{q}"] - 2))
    for size, v in sorted(by.items()):
        w, h = vlm.claude_resized(size, size)
        tok.setdefault(model, {})[str(size)] = dict(n=len(v), billed_image_tokens_min=min(v), billed_image_tokens_max=max(v),
                                                    formula=vlm.claude_tokens(w, h), resized=[w, h])
out["claude_image_tokens"] = tok
out["claude_count_tokens"] = list(ct.values())

# Cost per run, task and size, priced from each row's usage at batch rates (ledger.cost_of).
cost = {}
for model, rows in runs.items():
    if not rows or "usage" not in rows[0]:
        continue
    mid = "claude-sonnet-5-5" if model == "sonnet" else "claude-haiku-5-5"
    g = defaultdict(list)
    for r in rows:
        g[(r["task"], r["size"])].append(ledger.cost_of(mid, r["usage"], batch=True))
    for (task, size), cs in sorted(g.items()):
        cost.setdefault(model, {}).setdefault(task, {})[str(size)] = dict(n=len(cs), usd_per_1000=round(1000 * sum(cs) / len(cs), 4))
out["cost_batch"] = cost
tot = 0.0
for l in open("/home/amit/.cache/vidyalai-scratch/agents/lab/ledger.jsonl"):
    e = json.loads(l)
    if e["part"] == "v04":
        tot += e["cost"]
out["ledger_total_usd"] = round(tot, 4)

# Terse instruction: did the reply answer at once (<= 10 output tokens) or count aloud anyway?
ts = {}
for task in ("count", "dense"):
    for size in (448, 896):
        rs = [r for r in runs["haiku-terse"] if r["task"] == task and r["size"] == size]
        if rs:
            short = [r["ok"] for r in rs if r["usage"]["output_tokens"] <= 10]
            long_ = [r["ok"] for r in rs if r["usage"]["output_tokens"] > 10]
            ts[f"{task}@{size}"] = dict(answered_at_once=rate(short), counted_aloud=rate(long_))
out["terse_split"] = ts

out["tokens_local"] = json.load(open(f"{HERE}/tokens_local.json")) if os.path.exists(f"{HERE}/tokens_local.json") else []
out["timing"] = [{k: v for k, v in r.items()} for r in load("timing.jsonl")]
json.dump(out, open(f"{HERE}/results.json", "w"), indent=1)
print("wrote results.json", os.path.getsize(f"{HERE}/results.json"), "bytes")
