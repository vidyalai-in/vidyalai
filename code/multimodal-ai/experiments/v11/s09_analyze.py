"""Compile Part 11 results into runs/results.json (small, committable)."""
import sys, json, os
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np
import voice as V
D = "/home/amit/.cache/vidyalai-scratch/v11"
R = {}
ld = lambda p: json.load(open(f"{D}/runs/{p}")) if os.path.exists(f"{D}/runs/{p}") else None
# ---- Exp 1: codecs
codec = {}
for src in ["kokoro", "libri"]:
    js = [ld(f"codec_{src}_mimi.json"), ld(f"codec_{src}_dac.json"), ld(f"codec_{src}_mimi_dac.json")]
    st = ld(f"stoi_{src}.json") or {}
    cfgs, timing = {}, {}
    for j in js:
        if j:
            cfgs.update(j["configs"]); timing.update(j.get("timing_10s", {}))
    for k, c in cfgs.items():
        rows = np.array(c["err"])
        e = {"wer": V.boot_wer(rows), "n": len(rows), "words": int(rows[:, 1].sum())}
        if k in st:
            e["stoi"] = V.boot_ci(st[k]["stoi"], stat=np.mean)
        if k != "original":
            kind, nq = k.split("_"); nq = int(nq)
            fps, bits = (12.5, 11) if kind == "mimi" else (75, 10)
            e.update(codec=kind, nq=nq, tokens_per_s=fps * nq, kbps=fps * nq * bits / 1000)
        if k in timing:
            t = timing[k]
            e["enc_ms_10s"] = float(np.median(t["enc_ms"])); e["dec_ms_10s"] = float(np.median(t["dec_ms"]))
            e["enc_ms_range"] = [float(min(t["enc_ms"])), float(max(t["enc_ms"]))]
            e["dec_ms_range"] = [float(min(t["dec_ms"])), float(max(t["dec_ms"]))]
            e["gpu"] = {"clock_mhz": t["clock_mhz"], "power_w": t["power_w"]}
        codec[k] = e
    R[f"codec_{src}"] = codec; codec = {}
# ---- Exp 2: VAD
vad = ld("vad.json")
if vad:
    R["vad"] = {"chunk_ms_median": vad["chunk_ms_median"], "chunk_ms_p90": vad["chunk_ms_p90"], "thresholds": {}}
    for ms, rows in vad["runs"].items():
        d = [r["endpoint_delay"] for r in rows if r["endpoint_delay"] is not None]
        on = [r["onset_delay"] for r in rows if r["onset_delay"] is not None]
        k = sum(r["n_early_cuts"] > 0 for r in rows)
        R["vad"]["thresholds"][ms] = {"delay_p50": V.boot_ci(d), "delay_p90": V.boot_ci(d, stat=lambda x: np.percentile(x, 90)),
                                      "early_cut": V.wilson(k, len(rows)), "early_k": k, "n": len(rows),
                                      "onset_p50": V.boot_ci(on)}
    g = [x for it in vad["items"] for x in it["gaps"]]
    R["vad"]["pauses"] = {"n": len(g), "max_s": max(g), "median_s": float(np.median(g)), "turns_with_pause": sum(bool(it["gaps"]) for it in vad["items"])}
# ---- ASR
asr = ld("asr_turns.json")
if asr:
    R["asr"] = {k: {"time_p50": V.boot_ci(v["times"]), "time_p90": V.boot_ci(v["times"], stat=lambda x: np.percentile(x, 90)),
                    "wer": V.boot_wer(np.array(v["err"])), "gpu": v["gpu"], "peak_vram_mb": v["peak_vram_mb"]} for k, v in asr.items()}
# ---- LLM + TTS + totals
def llm_stats(path):
    recs = [json.loads(l) for l in open(f"{D}/runs/{path}") if '"ttft"' in l]
    out = {"n": len(recs)}
    for key in ("ttft", "t_first_sentence", "t_total"):
        x = [r[key] for r in recs]
        out[key] = {"p50": V.boot_ci(x), "p90": V.boot_ci(x, stat=lambda v: np.percentile(v, 90))}
    out["first_chars_median"] = float(np.median([r["first_chars"] for r in recs]))
    out["full_chars_median"] = float(np.median([r["full_chars"] for r in recs]))
    out["out_tokens_median"] = float(np.median([r["usage"]["output_tokens"] if "usage" in r else r.get("n_tokens", 0) for r in recs]))
    out["single_sentence_share"] = float(np.mean([r["first_chars"] >= r["full_chars"] - 1 for r in recs]))
    return recs, out
for name, path, tts in [("claude", "claude_stream.jsonl", "tts_claude_stream.json"), ("qwen", "qwen_stream.jsonl", "tts_qwen_stream.json")]:
    if not os.path.exists(f"{D}/runs/{path}"):
        continue
    recs, st = llm_stats(path)
    R[f"llm_{name}"] = st
    T = ld(tts)
    if not (T and vad and asr):
        continue
    vd = {r["turn"]: r["endpoint_delay"] for r in vad["runs"]["500"]}
    at = asr["openai/whisper-large-v3-turbo"]["times"]
    trow = {(r["turn"], r["rep"]): r for r in T["rows"]}
    stream, wait, waitstream = [], [], []
    comp = {"vad": [], "asr": [], "llm_first": [], "llm_total": [], "tts_first": [], "tts_full": []}
    for r in recs:
        t = trow[(r["turn"], r.get("rep", 0))]
        a, b = vd[r["turn"]], at[r["turn"]]
        stream.append(a + b + r["t_first_sentence"] + t["first_ttfa"])
        wait.append(a + b + r["t_total"] + t["full_total"])
        waitstream.append(a + b + r["t_total"] + t["full_ttfa"])
        for k, v in zip(comp, (a, b, r["t_first_sentence"], r["t_total"], t["first_ttfa"], t["full_total"])):
            comp[k].append(v)
    q90 = lambda v: np.percentile(v, 90)
    R[f"budget_{name}"] = {
        "n": len(recs),
        "streamed": {"p50": V.boot_ci(stream), "p90": V.boot_ci(stream, stat=q90)},
        "wait_full": {"p50": V.boot_ci(wait), "p90": V.boot_ci(wait, stat=q90)},
        "wait_then_stream_tts": {"p50": V.boot_ci(waitstream), "p90": V.boot_ci(waitstream, stat=q90)},
        "components": {k: {"p50": float(np.median(v)), "p90": float(q90(v))} for k, v in comp.items()},
        "tts_gpu": T["gpu"], "tts_peak_vram_mb": T["peak_vram_mb"],
        "saving_p50": V.boot_ci(np.array(wait) - np.array(stream)),
    }
q = ld("qwen_meta.json")
if q: R["qwen_meta"] = q
json.dump(R, open(f"{D}/runs/results.json", "w"), indent=1)
print(json.dumps(R, indent=1)[:6000])
