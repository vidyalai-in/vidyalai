"""Per-item comparison of our harness (GPT-2, FP32) with lm-eval-harness 0.4.13 samples."""
import json, glob
mine = json.load(open("post/results/gpt2_fp32.json"))["tasks"]
for task in ["hellaswag", "arc_easy", "piqa"]:
    f = glob.glob(f"post/lmeval/openai-community__gpt2/samples_{task}_*.jsonl")[0]
    theirs = sorted((json.loads(l) for l in open(f)), key=lambda d: d["doc_id"])
    ours = mine[f"{task}_0"]["items"]
    diffs, flips = [], 0
    for t, o in zip(theirs, ours):
        lt = [float(r[0][0]) for r in t["resps"]]
        diffs += [abs(a - b) for a, b in zip(lt, o["lls"])]
        flips += max(range(len(lt)), key=lambda c: lt[c]) != max(range(len(lt)), key=lambda c: o["lls"][c])
    print(task, len(theirs), "max |dLL|", f"{max(diffs):.2e}", "mean", f"{sum(diffs)/len(diffs):.2e}", "argmax flips", flips)
