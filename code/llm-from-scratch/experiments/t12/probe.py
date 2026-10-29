# Probe: the SFT model's pass rate on addition, by digits and prompt wording (greedy and sampled).
import sys, re, random, time, json
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
import torch
from sampling import generate_batch
from sft import chat_tokenizer, prompt_ids
from snapshots import load
SFT = '/home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt'
model, tok = load(SFT), chat_tokenizer()
NUM = re.compile(r'-?\d[\d,]*')
def last_int(s):
    m = NUM.findall(s)
    return int(m[-1].replace(',', '')) if m else None
rng = random.Random(0)
fmts = {"plain": "What is {a} + {b}?", "just": "What is {a} + {b}? Answer with just the number.",
        "bare": "{a} + {b} ="}
res = {}
for nd in (1, 2, 3):
    probs = [(rng.randint(10**(nd-1) if nd > 1 else 0, 10**nd - 1), rng.randint(10**(nd-1) if nd > 1 else 0, 10**nd - 1)) for _ in range(64)]
    for fname, f in fmts.items():
        P = [prompt_ids([{"role": "user", "content": f.format(a=a, b=b)}], tok) for a, b in probs]
        t = time.time()
        g = generate_batch(model, P, 96, 0.0)
        gacc = sum(last_int(tok.decode(ids)) == a + b for (ids, _), (a, b) in zip(g, probs)) / len(probs)
        gend = sum(e for _, e in g) / len(g)
        k = 16
        s = generate_batch(model, [p for p in P for _ in range(k)], 96, 1.0, seed=1)
        ok = [last_int(tok.decode(ids)) == a + b for (ids, _), (a, b) in zip(s, [p for p in probs for _ in range(k)])]
        p1 = sum(ok) / len(ok)
        passk = sum(any(ok[i*k:(i+1)*k]) for i in range(len(probs))) / len(probs)
        send = sum(e for _, e in s) / len(s); slen = sum(len(i) for i, _ in s) / len(s)
        print(f"{nd}d {fname:6} greedy {gacc:.3f} end {gend:.2f} | T=1 pass@1 {p1:.3f} pass@16 {passk:.3f} end {send:.2f} len {slen:.0f}  ({time.time()-t:.1f}s)", flush=True)
        print("   G:", repr(tok.decode(g[0][0])[:160]), "\n   S:", repr(tok.decode(s[0][0])[:160]), flush=True)
