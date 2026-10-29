# Before/after evaluation of one model on the addition task and beyond.
# usage: python evaluate.py <weights.pt> <tag> [--n 64] [--bench] [--chat] [--quick]
#  1. test sets (grpo.test_set): greedy accuracy (strict checker, and the last-integer checker), reply stats;
#     n samples per problem at temperature 1 for pass@k (k = 1, 2, 4, ... n), on 200 / 100 / 100 problems
#  2. --chat: fixed prompts through sft.chat (greedy, and temperature 0.8 / top-k 50 / seed 0), 200 tokens;
#     Part 10's 100 smol-smoltalk prompts, greedy and sampled, 256 tokens: ended share, length, repeats
#  3. --bench: Part 9's zero-shot benchmarks through evals.py
import sys, json, time, re, argparse
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/t11/scripts')
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
import torch
from grpo import test_set, sample, reward, pass_at_k
from sft import chat, chat_tokenizer, prompt_ids
from snapshots import load
p = argparse.ArgumentParser()
p.add_argument('path'); p.add_argument('tag')
p.add_argument('--n', type=int, default=64)
p.add_argument('--bench', action='store_true'); p.add_argument('--chat', action='store_true')
a = p.parse_args()
P = '/home/amit/.cache/vidyalai-scratch/t12/post/'
t0 = time.time()
model, tok = load(a.path), chat_tokenizer()
NUM = re.compile(r'\d[\d,]*')
def last_int(s):
    m = NUM.findall(s)
    return int(m[-1].replace(',', '')) if m else None
out = {"path": a.path, "tag": a.tag, "n": a.n}
tests = test_set()
samples_dump = {}
for split, size in (("test", 200), ("4digit", 100), ("5digit", 100)):
    probs = tests[split]
    prom = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
    g = sample(model, prom, 160, 0.0)
    gt = [tok.decode(i) for i, _ in g]
    strict = [reward(s, x + y) for s, (_, x, y) in zip(gt, probs)]
    lenient = [last_int(s) == x + y for s, (_, x, y) in zip(gt, probs)]
    res = {"greedy": sum(strict) / len(probs), "greedy_last_int": sum(lenient) / len(probs),
           "greedy_worked": sum(s.startswith("Add the digits") for s in gt) / len(gt),
           "greedy_ended": sum(e for _, e in g) / len(g), "greedy_len": sum(len(i) for i, _ in g) / len(g)}
    sub, sp = probs[:size], prom[:size]
    s = sample(model, [q for q in sp for _ in range(a.n)], 160, 1.0, seed=7)
    st = [tok.decode(i) for i, _ in s]
    ok = [reward(z, x + y) for z, (_, x, y) in zip(st, [q for q in sub for _ in range(a.n)])]
    wk = [z.startswith("Add the digits") for z in st]
    c = [int(sum(ok[i * a.n:(i + 1) * a.n])) for i in range(len(sub))]
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= a.n]
    res["pass_at_k"] = {k: sum(pass_at_k(a.n, ci, k) for ci in c) / len(c) for k in ks}
    res["correct_per_problem"] = c
    res["sampled_worked"] = sum(wk) / len(wk)
    res["sampled_acc_worked"] = sum(o for o, w in zip(ok, wk) if w) / max(1, sum(wk))
    res["sampled_acc_direct"] = sum(o for o, w in zip(ok, wk) if not w) / max(1, len(wk) - sum(wk))
    res["sampled_ended"] = sum(e for _, e in s) / len(s)
    res["sampled_len"] = sum(len(i) for i, _ in s) / len(s)
    res["sampled_distinct_ints"] = sum(len(set(re.findall(r'\d+', z))) for z in st) / len(st)
    out[split] = res
    samples_dump[split] = {"greedy": gt[:50], "sampled": st[:a.n * 3]}
    print(a.tag, split, {k: v for k, v in res.items() if k != "correct_per_problem"}, round(time.time() - t0), flush=True)
    print('   G:', repr(gt[0][:200]), flush=True)

if a.chat:
    from helpers import smol_prompts, sample_all, reply_stats
    U = lambda s: [{"role": "user", "content": s}]
    PROMPTS = {"capital": "What is the capital of France?", "sleep": "Give me three tips for sleeping better.",
               "math": "What is 17 + 25?", "add3": "What is 348 + 279?", "add4": "What is 4821 + 7765?",
               "sub": "What is 25 - 17?", "three": "Compute 12 + 34 + 56.",
               "word": "Tom has 17 apples and buys 25 more. How many apples does he have?"}
    out["fixed"] = {}
    for k, q in PROMPTS.items():
        gr = chat(model, tok, U(q), max_new=200)
        sa = chat(model, tok, U(q), max_new=200, temperature=0.8, top_k=50, seed=0)
        out["fixed"][k] = {"prompt": q, "greedy": gr, "sampled": sa}
        print(k, '|G|', gr[0][:200].replace('\n', ' '), gr[1:], '\n   |S|', sa[0][:200].replace('\n', ' '), sa[1:], flush=True)
    sp = smol_prompts(100)
    out["smol"] = {"greedy": reply_stats(sample_all(model, sp, 256, temperature=0)),
                   "sampled": reply_stats(sample_all(model, sp, 256, temperature=0.8, top_k=50, seed=0))}
    print(a.tag, 'smol', out["smol"], round(time.time() - t0), flush=True)

if a.bench:
    from evals import OurLM, load_task, score_task, summarize
    del model
    torch.cuda.empty_cache()
    lm = OurLM(a.path)
    out["bench"] = {}
    for task in ("hellaswag", "arc_easy", "piqa"):
        out["bench"][task] = summarize(score_task(lm, load_task(task)))
        print(task, out["bench"][task], round(time.time() - t0), flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(P + f'eval_{a.tag}.json', 'w'), indent=1)
json.dump(samples_dump, open(P + f'samples_{a.tag}.json', 'w'))
print('done', out["seconds"])
