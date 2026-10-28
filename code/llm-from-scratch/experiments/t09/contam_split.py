import json, sys
sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/fromscratch')
from evals import load_task
c = json.load(open('post/contamination.json'))
d = json.load(open('post/results/main88.json'))['tasks']
for t, norm in [('hellaswag', True), ('arc_easy', False), ('piqa', False)]:
    it = d[f'{t}_0']['items']; fl = c[t]['flagged']
    def acc(rs):
        h = [max(range(len(r['lls'])), key=lambda k: r['lls'][k] / (r['chars'][k] if norm else 1)) == r['gold'] for r in rs]
        return round(100 * sum(h) / len(h), 2), len(h)
    print(t, 'clean', acc([r for r, f in zip(it, fl) if not f]), 'flagged', acc([r for r, f in zip(it, fl) if f]))
    items = load_task(t)
    ex = [(items[i], c[t]['share_of_ngrams'][i]) for i in range(len(items)) if fl[i]]
    ex.sort(key=lambda x: -x[1])
    for e, s in ex[:3]:
        print('   ', round(s, 2), repr((e['query'] + ' ' + e['choices'][e['gold']])[:220]))
wt = c['wikitext']['share_of_ngrams']; print('wikitext share per article, top:', sorted([round(x,3) for x in wt], reverse=True)[:8])
