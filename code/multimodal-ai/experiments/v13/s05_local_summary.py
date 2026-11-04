"""Summaries of the local runs -> runs/results_local.json (small)."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import serving as S
D = "/home/amit/.cache/vidyalai-scratch/v13"
out = {"stages": [], "cache": {}, "batch": []}
for f in ("qwen", "smol"):
    for r in json.load(open(f"{D}/runs/local_stages_{f}.json")):
        row = {k: r[k] for k in ("model", "label", "w", "h", "longest_edge", "prompt_tokens", "image_tokens",
                                 "sm_clock_mhz", "power_w", "peak_vram_mb", "decode_ms_per_token")}
        for k in ("proc_ms", "enc_ms", "lm_prefill_ms", "full_prefill_ms", "gen1_ms", "gen33_ms"):
            row[k] = S.med(r[k]); row[k + "_range"] = [min(r[k]), max(r[k])]
        row["ttft_ms"] = row["proc_ms"] + row["enc_ms"] + row["lm_prefill_ms"]
        row["enc_share_of_ttft"] = row["enc_ms"] / row["ttft_ms"]
        out["stages"].append(row)
c = json.load(open(f"{D}/runs/local_cache.json"))
for v, reps in c["runs"].items():
    timed = reps[1:]
    per_turn = [[t["total_ms"] for t in rep] for rep in timed]
    arr = np.array(per_turn)
    out["cache"][v] = {"turn_median_ms": np.median(arr, 0).tolist(), "turn_min": arr.min(0).tolist(),
                       "turn_max": arr.max(0).tolist(), "conv_total_median_ms": float(np.median(arr.sum(1))),
                       "later_turns_median_ms": float(np.median(arr[:, 1:])),
                       "prompt_tokens": [t["prompt_tokens"] for t in timed[0]],
                       "reused_tokens": [t.get("reused_tokens") for t in timed[0]],
                       "replies": [t["reply"] for t in timed[0]]}
out["cache"]["compare"] = c["compare"]; out["cache"]["gpu"] = c["gpu"]; out["cache"]["answers"] = c["answers"]
out["cache"]["questions"] = c["questions"]; out["cache"]["image_tokens"] = c["image_tokens"]
for r in json.load(open(f"{D}/runs/local_batch.json")):
    out["batch"].append({k: (S.med(v) if isinstance(v, list) else v) for k, v in r.items()} |
                        {"enc_range": [min(r["enc_ms"]), max(r["enc_ms"])], "req_range": [min(r["req_ms"]), max(r["req_ms"])]})
json.dump(out, open(f"{D}/runs/results_local.json", "w"), indent=1)
for r in out["stages"]:
    print(r["model"].split("/")[1], r["label"], r["image_tokens"], {k: round(r[k], 1) for k in ("proc_ms", "enc_ms", "lm_prefill_ms", "full_prefill_ms", "ttft_ms", "decode_ms_per_token")}, round(r["enc_share_of_ttft"], 2), r["sm_clock_mhz"], round(r["power_w"]), round(r["peak_vram_mb"]))
for v in ("A", "B", "C"):
    x = out["cache"][v]; print(v, [round(t) for t in x["turn_median_ms"]], round(x["conv_total_median_ms"]), round(x["later_turns_median_ms"]), x["prompt_tokens"], x["reused_tokens"], x["replies"])
print(out["cache"]["answers"], out["cache"]["compare"], out["cache"]["gpu"], out["cache"]["image_tokens"])
for r in out["batch"]:
    print(r["batch"], round(r["enc_ms"]), round(r["images_per_s"], 2), round(r["req_ms"]), round(r["requests_per_s"], 2), round(r["out_tokens_per_s"], 1), round(r["req_peak_mb"]), r["sm_clock_mhz"], round(r["power_w"]))
