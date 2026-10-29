# Before/after for the sequence-distillation models (88M, chat template from Part 10):
# held-out reply loss on three sets, reply statistics on 200 held-out prompts (greedy and sampled,
# batched with Part 11's sampling.generate_batch), fixed prompts, teacher-identity strings, Part 9 benchmarks.
# usage: python eval_sft.py <tag> <weights.pt> [--bench]
import sys, json, time, re, statistics, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
import sft
from sft import Conversations, response_loss, chat, chat_tokenizer, prompt_ids
from sampling import generate_batch
from snapshots import load
tag, path = sys.argv[1], sys.argv[2]
D = '/home/amit/.cache/vidyalai-scratch/t13/data'
t0 = time.time(); out = {"tag": tag, "path": path}
model, tok = load(path), chat_tokenizer()
out["loss_part10_val"] = response_loss(model, Conversations("val"))           # 2,000 multi-turn, original replies
sft.DATA = f"{D}/evalsets"
out["loss_first_orig"] = response_loss(model, Conversations("orig"))          # first exchange, original replies
out["loss_first_qwen"] = response_loss(model, Conversations("qwen"))          # same prompts, teacher replies
print(tag, {k: round(v["all"], 4) for k, v in out.items() if k.startswith("loss")}, flush=True)
ids = [prompt_ids(p["prompt"], tok) for p in json.load(open(f"{D}/prompts.json"))["val"]]
ids = [x for x in ids if len(x) <= 512][:200]                  # room for a 256-token reply
def stats(temp, top_k):
    rows = []
    for b in range(0, len(ids), 50):
        rows += generate_batch(model, ids[b:b + 50], max_new=256, temperature=temp, top_k=top_k, seed=b)
    res = []
    for body, ended in rows:
        g = [tuple(body[j:j + 4]) for j in range(len(body) - 3)]
        res.append({"ended": ended, "len": len(body), "rep4": 1 - len(set(g)) / max(1, len(g)),
                    "text": tok.decode(body)})
    return {"n": len(res), "ended": sum(r["ended"] for r in res) / len(res),
            "len_median": statistics.median(r["len"] for r in res),
            "len_ended_median": statistics.median([r["len"] for r in res if r["ended"]] or [0]),
            "rep4_mean": statistics.mean(r["rep4"] for r in res),
            "identity": sum(bool(re.search(r"Qwen|Alibaba", r["text"])) for r in res),
            "cjk": sum(bool(re.search(r"[一-鿿]", r["text"])) for r in res),
            "markdown": sum(bool(re.search(r"\*\*|^#+ |^\d+\. ", r["text"], re.M)) for r in res) / len(res)}, [r["text"] for r in res]
out["greedy"], g_txt = stats(0.0, None)
out["sampled"], s_txt = stats(0.8, 50)
print("gen", out["greedy"], out["sampled"], round(time.time() - t0), flush=True)
U = lambda s: [{"role": "user", "content": s}]
FIXED = ["What is the capital of France?", "Why is the sky blue?", "Give me three tips for sleeping better.",
         "Explain photosynthesis to a ten-year-old in two sentences.", "What is 17 + 25?",
         "What was the weather like in Pune yesterday?", "What is my name?", "Who are you?",
         "Who created you?"]
out["fixed"] = {q: {"greedy": chat(model, tok, U(q), max_new=200),
                    "sampled": chat(model, tok, U(q), max_new=200, temperature=0.8, top_k=50, seed=0)} for q in FIXED}
for q, r in out["fixed"].items():
    print(q, "|G|", r["greedy"][0][:200].replace("\n", " "), r["greedy"][1:], "|S|", r["sampled"][0][:200].replace("\n", " "), flush=True)
if "--bench" in sys.argv:
    from evals import OurLM, load_task, score_task, summarize
    lm = OurLM(path)
    out["bench"] = {t: summarize(score_task(lm, load_task(t))) for t in ("hellaswag", "arc_easy", "piqa")}
    print(out["bench"], flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(f"/home/amit/.cache/vidyalai-scratch/t13/post/sft_{tag}.json", "w"), indent=1)
json.dump({"greedy": g_txt, "sampled": s_txt}, open(f"{D}/gen_{tag}.json", "w"))
print("done", out["seconds"])
