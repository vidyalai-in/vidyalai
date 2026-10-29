"""Replay a torch.cuda.memory snapshot from bench.py and break memory down by what allocated it (Part 14).

Every allocation recorded during the timed step carries Python frames: the bench.py line (inputs, forward,
backward, clipping, optimizer) and, for compiled code, the model.py line Inductor generated it from, plus
the generated line itself, which gives shape and dtype. Memory allocated before recording started is the
weights and AdamW state (the run's own counters give its size).
    python snap.py base_c16
"""
import collections
import json
import linecache
import pickle
import re
import sys

RUNS = "/home/amit/.cache/vidyalai-scratch/t14/runs"
tag = sys.argv[1]
snap = pickle.load(open(f"{RUNS}/{tag}.snapshot.pickle", "rb"))
run = json.load(open(f"{RUNS}/{tag}.json"))
grad_ptrs = {int(k) for k in run.get("grad_ptrs", {})}
BENCH = {100: "inputs", 102: "forward", 105: "backward", 109: "clip", 113: "optimizer"}
MODEL_LINES = {54: "norm", 55: "norm", 56: "norm", 89: "wq", 90: "wk", 91: "wv", 92: "rope", 93: "attention",
               94: "wo", 105: "swiglu", 119: "residual", 120: "residual", 154: "embedding", 156: "blocks",
               157: "final norm + logits", 160: "loss"}
SHAPE = re.compile(r"empty_strided_cuda\(\(([^)]*)\),\s*\([^)]*\),\s*torch\.(\w+)\)")


def describe(frames):
    phase, src, gen = "other", None, None
    for f in frames:
        if f["filename"].endswith("bench.py") and f["line"] in BENCH:
            phase = BENCH[f["line"]]
        if f["filename"].endswith("fromscratch/model.py") and src is None:
            src = MODEL_LINES.get(f["line"], f"model.py:{f['line']}")
        if "torchinductor" in f["filename"] and gen is None:
            m = SHAPE.search(linecache.getline(f["filename"], f["line"]))
            gen = (m.group(2), m.group(1).replace(" ", "")) if m else None
    return phase, src, gen


trace = snap["device_traces"][0]
mb, cfg = run["args"]["mb"], run["config"]
tokens = mb * 1024
d, f, V = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]
q, kv = cfg["n_head"] * 64, cfg["n_kv_head"] * 64
WEIGHT_COPIES = {2 * d * q, 2 * d * kv, 2 * d * f, 2 * V * d}       # BF16 casts of each weight shape


def category(addr, size, phase, src, gen):
    if addr in grad_ptrs:
        return "gradients"
    if phase == "forward":
        if gen:
            dims = [int(x) for x in gen[1].split(",") if x]
            is_act = not dims or mb * 1024 in (dims[0], dims[0] * (dims[1] if len(dims) > 1 else 1)) or dims[:2] == [mb, 1024] \
                or (len(dims) > 2 and dims[0] == mb)
        else:
            is_act = size not in WEIGHT_COPIES
        if not is_act:
            return "BF16 weight copies"
        return "output layer and loss" if src in ("final norm + logits", "loss") else "saved activations (blocks)"
    if phase == "inputs":
        return "inputs"
    return "backward temporaries"


base = run["phases"][0][1]                       # allocated when recording started: weights + AdamW state
live, total, peak, peak_live = {}, base, base, {}
fwd_runs, prev, before_bwd = 0, None, None
for e in trace:
    if e["action"] == "alloc":
        ph, src, gen = describe(e["frames"])
        if ph == "forward" and prev != "forward":
            fwd_runs += 1
        if fwd_runs == 2 and ph != "forward" and prev == "forward" and before_bwd is None:
            before_bwd = (total, dict(live))      # 2nd micro-batch's forward done: gradients exist, activations full
        prev = ph
        live[e["addr"]] = (e["size"], category(e["addr"], e["size"], ph, src, gen), (ph, src, gen))
        total += e["size"]
        if total > peak:
            peak, peak_live = total, dict(live)
    elif e["action"] == "free_completed" and e["addr"] in live:
        total -= live.pop(e["addr"])[0]

G = 2 ** 30
w = run["after_weights"]
res = {"tag": tag, "weights": w, "adamw_and_other_state": base - w, "peak_replayed": peak,
       "peak_counter": run["peak_alloc"], "reserved_peak": run["peak_reserved"]}
for name, (tot, lv) in (("before 2nd backward", before_bwd), ("peak", (peak, peak_live))):
    cat, detail = collections.Counter(), collections.Counter()
    for size, c, (ph, src, gen) in lv.values():
        cat[c] += size
        detail[(c, src, gen[0] if gen else "", gen[1] if gen else "")] += size
    print(f"--- {name}: {tot / G:.3f} GiB = weights {w / G:.3f} + AdamW etc. {(base - w) / G:.3f} + live during the step:")
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {v / G:7.3f} GiB  {k}  ({v / tokens:,.0f} bytes per token)")
    res[name] = {"total": tot, "categories": dict(cat),
                 "detail": [[*k, v] for k, v in sorted(detail.items(), key=lambda kv: -kv[1]) if v > 2 ** 20]}
print(f"replayed peak {peak / G:.3f} GiB vs max_memory_allocated {run['peak_alloc'] / G:.3f} GiB")
json.dump(res, open(f"/home/amit/.cache/vidyalai-scratch/t14/post/snap_{tag}.json", "w"), indent=1)
