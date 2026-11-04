import json, numpy as np
R='/home/amit/.cache/vidyalai-scratch/v05/runs'
D={}
for k in ('linear','cosine'):
    s=json.load(open(f'{R}/fm_{k}/state.json'))
    lg=np.array(s['log'])   # every 50 steps
    w=5; loss=[[int(lg[i+w-1,0]), round(float(lg[i:i+w,1].mean()),4)] for i in range(0,len(lg)-w+1,w)]
    fd=[[e['step'], round(e['train_s'],1), round(e['fd'],2)] for e in s['evals']]
    D[k]={'loss':loss,'fd':fd}
import sys
extra = json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
for k,v in extra.items(): D[k]['fd'].append(v)
xmax = max(p[0] for v in D.values() for p in v['loss'])
s=open('gen_train.tpl').read().replace('__DATA__', json.dumps(D,separators=(',',':')))
s=s.replace('__XMAX_STEPS__', str(int(np.ceil(xmax/2500)*2500))).replace('__LY0__','0.02').replace('__LY1__','0.12')
s=s.replace('__LTICKS__','(0.02, 0.04, 0.06, 0.08, 0.1, 0.12)').replace('__XTICKS_STEPS__', str(tuple(range(0,int(np.ceil(xmax/2500)*2500)+1,2500))))
open('/home/amit/AlpsCode/ai/vidyalai/diagrams/v05-fm-training.py','w').write(s)
print({k:v['fd'] for k,v in D.items()})
