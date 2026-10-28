import numpy as np, json, collections, sys
sys.path.insert(0,'.')
D='data/smol-smoltalk'
out={}
for s in ('train','val'):
    ids=np.load(f'{D}/{s}_ids.npy'); m=np.load(f'{D}/{s}_mask.npy'); off=np.load(f'{D}/{s}_offsets.npy')
    src=json.load(open(f'{D}/{s}_sources.json'))
    L=np.diff(off)-1; T=np.add.reduceat(m.astype(np.int64), off[:-1])
    nassist=np.add.reduceat((ids==3).astype(np.int64), off[:-1])
    r={'conversations':len(L),'tokens':int(off[-1]),'input_tokens':int(L.sum()),'targets':int(m.sum()),
       'prompt_tokens':int(L.sum()-m.sum()),'mean_len':float(L.mean()),'median_len':float(np.median(L)),
       'p90_len':float(np.percentile(L,90)),'share_targets':float(m.sum()/L.sum()),
       'multi_turn':float((nassist>1).mean()), 'mean_assistant_turns': float(nassist.mean())}
    by=collections.defaultdict(lambda:[0,0,0])
    for i,sname in enumerate(src): by[sname][0]+=1; by[sname][1]+=int(L[i]); by[sname][2]+=int(T[i])
    r['by_source']={k:v for k,v in sorted(by.items(), key=lambda kv:-kv[1][0])}
    out[s]=r
print(json.dumps(out,indent=1))
json.dump(out,open('../t10/post/data_stats.json','w'),indent=1)
