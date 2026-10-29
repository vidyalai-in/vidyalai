# Collect the small result files into post/ (committable, well under 400 KB in total).
import json, os, shutil
R = "/home/amit/.cache/vidyalai-scratch/p03/runs"; P = "/home/amit/.cache/vidyalai-scratch/p03/post"
os.makedirs(P, exist_ok=True)
pos = {"gpt2": json.load(open(f"{R}/pos_gpt2_all.json"))}
a, b = json.load(open(f"{R}/pos_qwen_a.json")), json.load(open(f"{R}/pos_qwen_b.json"))
a["layers"].update(b["layers"]); pos["qwen"] = a
json.dump(pos, open(f"{P}/pos_probes.json", "w"), indent=1)
json.dump({w: json.load(open(f"{R}/truth_{w}.json")) for w in ["gpt2", "qwen"]}, open(f"{P}/truth_probes.json", "w"), indent=1)
shutil.copy(f"{R}/transfer_auc.json", f"{P}/truth_template_transfer_auc.json")
json.dump({"gpt2": json.load(open(f"{R}/nudge_gpt2_fine.json")), "qwen_coarse": json.load(open(f"{R}/nudge_qwen.json")),
           "qwen_fine": json.load(open(f"{R}/nudge_qwen_fine.json"))}, open(f"{P}/nudge.json", "w"), indent=1)
for f in sorted(os.listdir(P)):
    print(f, os.path.getsize(f"{P}/{f}"))
