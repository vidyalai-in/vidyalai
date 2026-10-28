# Reply statistics with sampling (temperature 0.8, top-k 50, seed = prompt index) on the same 100 held-out prompts.
import sys, json, statistics, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import Conversations, END
from snapshots import load
val = Conversations("val")
out = {}
for tag in sys.argv[1:]:
    model, st = load(f'/home/amit/.cache/vidyalai-scratch/t10/runs/{tag}/model_final.pt'), []
    for i in range(100):
        ids = val.ids[val.off[i]:val.off[i + 1]].tolist()
        cut = ids.index(3)
        idx = torch.tensor([ids[:cut + 1]], device="cuda")
        torch.manual_seed(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.generate(idx, 256, temperature=0.8, top_k=50, eos_id=END)[0, cut + 1:].tolist()
        ended = o[-1] == END; body = o[:-1] if ended else o
        grams = [tuple(body[j:j + 4]) for j in range(len(body) - 3)]
        st.append({"ended": ended, "len": len(body), "rep4": 1 - len(set(grams)) / max(1, len(grams))})
    out[tag] = {"ended": sum(s["ended"] for s in st) / 100, "len_ended_median": statistics.median([s["len"] for s in st if s["ended"]]),
                "rep4_mean": statistics.mean(s["rep4"] for s in st)}
    print(tag, out[tag], flush=True)
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t10/post/sampled_gen.json', 'w'), indent=1)
