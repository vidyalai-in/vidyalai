# Summarize bench.py runs: results plus clocks/power from nvidia-smi during the timed steps only.
import datetime, glob, json, os, statistics as st, sys

D = "/home/amit/.cache/vidyalai-scratch/t06/logs/"


def smi_window(name, t0, t1):
    rows = []
    for l in open(D + name + ".smi.csv"):
        f = [x.strip() for x in l.split(",")]
        try:
            ts = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            rows.append(dict(sm=int(f[1].split()[0]), mem=int(f[2].split()[0]), pw=float(f[3].split()[0]),
                             lim=float(f[4].split()[0]), temp=int(f[5])))
    if not rows:
        return {}
    sm = [r["sm"] for r in rows]
    return dict(sm_med=round(st.median(sm)), sm_min=min(sm), sm_max=max(sm),
                mem=sorted(set(r["mem"] for r in rows)), pw_mean=round(st.mean(r["pw"] for r in rows), 1),
                lim=(min(r["lim"] for r in rows), max(r["lim"] for r in rows)),
                temp=(min(r["temp"] for r in rows), max(r["temp"] for r in rows)), n=len(rows))


def summarize(name):
    steps, res = [], None
    for l in open(D + name + ".out"):
        if l.startswith('{"step"'):
            steps.append(json.loads(l))
        if l.startswith("RESULT "):
            res = json.loads(l[7:])
    if res is None:
        return None
    out = dict(name=name, mb=res["mb"], accum=res["accum"], tok_s=round(res["tok_s"]),
               mfu=round(100 * res["mfu"], 1), mfu_spec=round(100 * res["tflops"] / 29.1, 1),
               tflops=round(res["tflops"], 2), peak=round(res["peak_alloc_GiB"], 2),
               reserved=round(res["peak_reserved_GiB"], 2), first_micro_s=res["first_micro_s"],
               warmup_s=res["warmup_s"])
    if steps:
        tps = [s["tok_s"] for s in steps]
        out.update(step_s=round(res["step_s_median"], 3), steps=len(steps), tok_s_min=min(tps), tok_s_max=max(tps),
                   opt_ms=round(res["opt_ms_median"], 1), micro_ms=res["micro_ms_median"],
                   first_micro_ms=[round(x, 1) for x in res["first_micro_ms"]],
                   data_ms=[s["data_ms"] for s in steps], loss=steps[-1]["loss"])
        out.update(smi_window(name, steps[0]["t"] - steps[0]["s"], steps[-1]["t"]))
    else:
        out["micro_s_median"] = round(res["micro_s_median"], 4)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(D + "*.out"))
    for n in names:
        try:
            r = summarize(n)
        except Exception as e:                  # noqa: BLE001
            r = dict(name=n, error=str(e))
        if r:
            print(json.dumps(r))
