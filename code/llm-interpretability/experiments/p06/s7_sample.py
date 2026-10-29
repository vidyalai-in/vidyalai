"""Print 30 random sparse neurons (seed 0) with their top token types per domain, for manual rating."""
import json, random, torch
RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
z = torch.load(RUN + "s6_counts.pt"); e = torch.load(RUN + "s7_events.pt")
nd, L, H, V = 4, e["L"], e["H"], e["V"]; types = e["types"]; DOMS = z["doms"]
keys, cnt = e["keys"], e["counts"].long()
g_of = keys // V // nd
tot = torch.zeros(L * H, dtype=torch.long).index_add_(0, g_of, cnt)
ntok = 4 * 400 * 127
sparse = ((tot >= 20) & (tot < 0.01 * ntok)).nonzero().flatten().tolist()
random.seed(0); pick = sorted(random.sample(sparse, 30))
out = []
for g in pick:
    m = g_of == g; kk, cc = keys[m], cnt[m]
    row = dict(neuron=f"{g // H}.{g % H}", fires=int(tot[g]), top={})
    for k, d in enumerate(DOMS):
        mm = (kk // V) % nd == k
        s = int(cc[mm].sum())
        o = cc[mm].argsort(descending=True)[:6]
        row["top"][d] = (s, [(types[int(kk[mm][j] % V)], int(cc[mm][j])) for j in o])
    out.append(row)
    print(row["neuron"], row["fires"])
    for d in DOMS: print("   ", d, row["top"][d][0], row["top"][d][1])
json.dump(out, open(RUN + "s7_sample30.json", "w"), ensure_ascii=False, indent=1)
