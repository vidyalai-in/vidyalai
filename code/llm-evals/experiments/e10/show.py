import json,sys
sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/evals')
from lab import safety_evals as S
X={r['id']:r for r in S.load_xstest()}
smp=json.load(open('/home/amit/.cache/vidyalai-scratch/e10/runs/hand_sample.json'))
a,b=int(sys.argv[1]),int(sys.argv[2])
cache={}
for i,s in enumerate(smp[a:b],a):
    tag,k=s['key'].split('/')
    if tag not in cache: cache[tag]={json.loads(l)['id']:json.loads(l) for l in open(f'/home/amit/.cache/vidyalai-scratch/e10/runs/xstest_{tag}.jsonl')}
    t=cache[tag][k]['text']
    print(f"#{i} {s['key']} [{s['label']}/{s['type']}] Q: {X[int(k[3:])]['prompt']}\nA: {t[:600].replace(chr(10),' / ')}\n")
