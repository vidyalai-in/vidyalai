"""Is the cached, batched path exact up to bf16 noise? Compare one item four ways."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import mcq
mid = sys.argv[1]; dt = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=dt, device_map="cuda").eval()
sc = mcq.LogProbScorer(model, tok)
I = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/mmlu-1140.json"))
S = json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e04-shots.json"))
fmt = mcq.FORMATS["mmlu"]
its = [I[20 + k] for k in range(3)]
parts = [mcq.prompt(fmt, it, S[it["subject"]]["dev"]) for it in its]
with torch.no_grad():
    full = sc.bos + sc.enc(parts[0][0] + parts[0][1])
    h = model.model(input_ids=torch.tensor([full], device="cuda")).last_hidden_state
    a = sc._logprobs_at(h, [(0, len(full) - 1)])[0]
    ids = [sc.enc(parts[0][1] + fmt.target(c))[-1] for c in range(4)]
    print("plain        ", [round(a[t].item(), 4) for t in ids])
    print("cached b=1   ", [round(x, 4) for x in sc.label_scores(parts[0][0], [parts[0][1]], fmt)[0]])
    print("cached b=3   ", [round(x, 4) for x in sc.label_scores(parts[0][0], [p[1] for p in parts], fmt)[0]])
    sc2 = mcq.LogProbScorer(model, tok); 
    print("no cache b=1 ", [round(x, 4) for x in sc2.label_scores("", [parts[0][0] + parts[0][1]], fmt)[0]])
