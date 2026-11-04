import json, math, collections
rows=[json.loads(l) for l in open('/home/amit/.cache/vidyalai-scratch/v01/runs/dots_batch.jsonl')]
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
by=collections.defaultdict(list)
for r in rows: by[r['size']].append(r)
out={}
for s in ['150x100','300x200','600x400','1200x800','2400x1600']:
    rs=by[s]; n=len(rs); ok=[r for r in rs if r['type']=='succeeded']
    k=sum(1 for r in ok if r['parsed']==r['n']); unp=sum(1 for r in ok if r['parsed'] is None)
    err=[r['parsed']-r['n'] for r in ok if r['parsed'] is not None]
    mae=sum(abs(e) for e in err)/len(err)
    lo,hi=wilson(k,n)
    out[s]=dict(n=n,correct=k,acc=k/n,ci=[lo,hi],unparsed=unp,mae=mae,mean_err=sum(err)/len(err),tokens=sorted(set(r['input_tokens'] for r in ok))[:3],cost=sum(r['cost'] for r in ok))
    print(s,out[s])
# paired: 150 vs 1200
import itertools
idx={ (r['item'],r['size']):r for r in rows}
for a,b in [('150x100','1200x800'),('300x200','1200x800'),('1200x800','2400x1600')]:
    d=[(idx[(i,a)]['parsed']==idx[(i,a)]['n'])-(idx[(i,b)]['parsed']==idx[(i,b)]['n']) for i in range(40)]
    print(a,b,'only a right',d.count(1),'only b right',d.count(-1))
json.dump(out,open('/home/amit/.cache/vidyalai-scratch/v01/runs/dots_summary.json','w'),indent=1)
