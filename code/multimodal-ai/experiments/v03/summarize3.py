import json, numpy as np, collections
R=[json.loads(l) for l in open('exp3-runs.jsonl')]
g=collections.defaultdict(list)
for r in R: g[(r['tag'],r['loss'],r['B'],r['tau'])].append(r)
out=[]
for k,v in g.items():
    if k[0] in ('p','pilot'): continue
    f=lambda key: [x[key] for x in v]
    se=lambda a: np.array(a)
    row=dict(tag=k[0],loss=k[1],B=k[2],tau=k[3],n=len(v),steps=v[0]['steps'],
      seen=[float(np.mean(f2)) for f2 in [[x['te']['top1'] for x in v]]][0], seen_min=min(x['te']['top1'] for x in v), seen_max=max(x['te']['top1'] for x in v),
      unseen=float(np.mean([x['un']['top1'] for x in v])), unseen_min=min(x['un']['top1'] for x in v), unseen_max=max(x['un']['top1'] for x in v),
      evloss=float(np.mean(f('eval_loss_b256'))), gap=float(np.mean(f('gap'))), gap_min=min(f('gap')), gap_max=max(f('gap')),
      scale=float(np.mean(f('final_scale'))), wall=float(np.median(f('wall_s'))), cos_match=float(np.mean(f('cos_match'))))
    out.append(row)
    print(f"{row['tag']:6s} {row['loss']:7s} B={row['B']:5d} tau={row['tau']:8s} steps={row['steps']:5d} seen {row['seen']*100:5.1f} [{row['seen_min']*100:.1f},{row['seen_max']*100:.1f}] unseen {row['unseen']*100:5.1f} [{row['unseen_min']*100:.1f},{row['unseen_max']*100:.1f}] evloss {row['evloss']:.3f} gap {row['gap']:.3f} [{row['gap_min']:.3f},{row['gap_max']:.3f}] scale {row['scale']:.1f} cos {row['cos_match']:.3f} wall {row['wall']:.1f}")
json.dump(out,open('exp3-summary.json','w'),indent=1)
