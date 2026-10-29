# How often do replies pad to extra columns and answer with a leading zero ("The answer is 042.")?
# Greedy on the 500 held-out problems and the 200 4-digit ones; also accuracy if leading zeros were rejected.
import sys, re, json
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from grpo import test_set, sample, reward
from sft import chat_tokenizer, prompt_ids
from snapshots import load
tok, tests, out = chat_tokenizer(), test_set(), {}
STEP = re.compile(r"(\d) \+ (\d)(?: \+ (\d))? = (\d+), write (\d)")
for tag, path in (("warm_mixed", "warm_mixed"), ("grpo_mixed", "grpo_mixed")):
    model = load(f'/home/amit/.cache/vidyalai-scratch/t12/runs/{path}/model_final.pt')
    res = {}
    for split in ("test", "4digit"):
        probs = tests[split]
        P = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
        texts = [tok.decode(i) for i, _ in sample(model, P, 160, 0.0)]
        lead = [bool(re.search(r"The answer is 0\d", t)) for t in texts]
        ok = [reward(t, a + b) for t, (_, a, b) in zip(texts, probs)]
        cols = [len(STEP.findall(t)) for t in texts]
        need = [max(len(str(a)), len(str(b))) for _, a, b in probs]
        res[split] = {"acc": sum(ok) / len(ok), "leading_zero": sum(lead) / len(lead),
                      "acc_no_leading_zero": sum(o and not l for o, l in zip(ok, lead)) / len(ok),
                      "padded": sum(c > n for c, n, t in zip(cols, need, texts) if t.startswith("Add")) / len(texts),
                      "cols_hist": {c: cols.count(c) for c in sorted(set(cols))},
                      "worked": sum(t.startswith("Add") for t in texts) / len(texts)}
        print(tag, split, res[split], flush=True)
    out[tag] = res
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t12/post/padding_mixed.json', 'w'), indent=1)
