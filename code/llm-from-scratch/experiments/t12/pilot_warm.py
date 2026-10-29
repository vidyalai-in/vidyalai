# Warm-start pilots: SFT on n synthetic problems (half direct, half worked), then pass rates.
import sys, time, json
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
import torch
import grpo
from grpo import get_args, warmstart, test_set, sample, reward, pass_at_k
from sft import chat_tokenizer, prompt_ids
from snapshots import load
R = '/home/amit/.cache/vidyalai-scratch/t12/runs/'
tok, tests = chat_tokenizer(), test_set()
out = {}
for n in [int(v) for v in sys.argv[1].split(',')]:
    lr = sys.argv[2] if len(sys.argv) > 2 else '3e-4'
    t = time.time()
    warmstart(get_args(['warmstart', '--out', R + f'warm_n{n}_lr{lr}', '--n', str(n), '--lr', lr, '--warmup', '5', '--fused']))
    tw = time.time() - t
    m = load(R + f'warm_n{n}_lr{lr}/model_final.pt')
    res = {"warm_s": round(tw, 1)}
    for split, probs, k in (("test", tests["test"][:200], 8), ("4digit", tests["4digit"][:100], 8)):
        P = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
        t = time.time()
        g = sample(m, P, 160, 0.0)
        tg = time.time() - t
        gt = [tok.decode(i) for i, _ in g]
        gok = [reward(s, x + y) for s, (_, x, y) in zip(gt, probs)]
        t = time.time()
        s = sample(m, [p for p in P for _ in range(k)], 160, 1.0, seed=1)
        ts = time.time() - t
        st = [tok.decode(i) for i, _ in s]
        ok = [reward(z, x + y) for z, (_, x, y) in zip(st, [p for p in probs for _ in range(k)])]
        wk = [z.startswith("Add the digits") for z in st]
        c = [int(sum(ok[i*k:(i+1)*k])) for i in range(len(probs))]
        acc_w = sum(o for o, w in zip(ok, wk) if w) / max(1, sum(wk)); acc_d = sum(o for o, w in zip(ok, wk) if not w) / max(1, len(wk) - sum(wk))
        res[split] = {"greedy": sum(gok) / len(gok), "greedy_worked": sum(z.startswith("Add the digits") for z in gt) / len(gt),
                      "pass1": sum(ok) / len(ok), f"pass{k}": sum(pass_at_k(k, ci, k) for ci in c) / len(c),
                      "zero_groups": sum(ci == 0 for ci in c) / len(c), "all_groups": sum(ci == k for ci in c) / len(c),
                      "worked_frac": sum(wk) / len(wk), "acc_worked": acc_w, "acc_direct": acc_d,
                      "ended": sum(e for _, e in s) / len(s), "len": sum(len(i) for i, _ in s) / len(s),
                      "t_greedy": round(tg, 1), "t_sample": round(ts, 1), "rows": len(s)}
        print(n, lr, split, json.dumps(res[split]), flush=True)
        print('   G:', repr(gt[0][:150]), '| S:', repr(st[1][:150]), flush=True)
    out[f'{n}_{lr}'] = res
    del m; torch.cuda.empty_cache()
json.dump(out, open(f'/home/amit/.cache/vidyalai-scratch/t12/post/pilot_warm_{sys.argv[1]}_{sys.argv[2] if len(sys.argv) > 2 else "3e-4"}.json', 'w'), indent=1)
