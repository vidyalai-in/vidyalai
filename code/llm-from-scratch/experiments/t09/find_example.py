import json, sys, os
sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/fromscratch')
from evals import load_task
d=json.load(open('post/results/main88.json'))
items=load_task('arc_easy')
res=d['tasks']['arc_easy_0']['items']
cnt=0
for i,(it,r) in enumerate(zip(items,res)):
    lls=r['lls']; g=r['gold']
    raw=max(range(len(lls)),key=lambda c:lls[c]); norm=max(range(len(lls)),key=lambda c:lls[c]/r['chars'][c])
    if (raw!=norm) and len(it['query'])<80 and len(lls)==4:
        print(i, repr(it['query']), it['choices'], 'gold',g,'raw',raw,'norm',norm, [round(x,2) for x in lls]); cnt+=1
    if cnt>15: break
