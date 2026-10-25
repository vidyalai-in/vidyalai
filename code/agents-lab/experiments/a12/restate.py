import json, sys, glob
sys.path.insert(0,'.')
from chain import nxt
from collections import Counter
c=Counter()
rows={json.loads(l)['id']:json.loads(l) for l in open('runs/chain-off.jsonl')}
for rid,r in rows.items():
    if r['level']!='xhard' or r['seed']>900: continue
    ev=[json.loads(l) for l in open(f'runs/tx/chain-off/{rid}.jsonl')]
    texts=[ ''.join(b.get('text','') for b in e['content'] if b['type']=='text') for e in ev if e['event']=='step']
    # step i's arithmetic is in model call i+1
    for i,ok in enumerate(r['steps']):
        t=texts[i+1] if i+1<len(texts) else ''
        restated = t.strip().startswith('Record') or 'm = ' in t
        pos = 'early' if i<2 else 'late'
        c[(pos,restated,ok)]+=1
for pos in ('early','late'):
    for rs in (True,False):
        ok=c[(pos,rs,1)]; bad=c[(pos,rs,0)]
        if ok+bad: print(pos,'restates' if rs else 'no restate', bad,'/',ok+bad, f'{bad/(ok+bad):.1%}')
