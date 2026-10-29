import json, numpy as np
R='/home/amit/.cache/vidyalai-scratch/p04/runs/'
rep=json.load(open(R+'s2_gpt2_rep.json'))
try: k4=json.load(open(R+'s3_gpt2_rep.json'))['top']['4'][0]
except Exception: k4=None
ks=[0,1,2,3,4,5,8,12]
top=[rep['rep_clean'][0]]+[rep['rep_top'][str(k)][0] if k!=4 else k4 for k in ks[1:]]
rnd={k:rep['rep_rand'][str(k)] for k in [1,2,3,5,8,12]}
nat=json.load(open(R+'s2_gpt2_nat.json'))['nat']
edges=np.unique(np.round(np.geomspace(1,513,22)).astype(int))
def binned(c):
    c=np.array(c); return [[float(np.sqrt(a*(b-1))), float(c[a-1:b-1].mean())] for a,b in zip(edges[:-1],edges[1:])]
clean=binned(nat['clean']['curve']); t12=binned(nat['top12']['curve'])
r12=binned(np.mean([nat[f'rand12_{j}']['curve'] for j in range(5)],0))
out={'ks':ks,'top':top,'rand':{k:[v['p5'],v['p50'],v['p95']] for k,v in rnd.items()},'clean':clean,'top12':t12,'rand12':r12}
print(json.dumps(out))
