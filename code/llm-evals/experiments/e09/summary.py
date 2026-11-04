import json
d=json.load(open('/home/amit/.cache/vidyalai-scratch/e09/runs/e09-results.json'))
def c(x): return f"{x['k']}/{x['n']} {100*x['rate']:.0f}% [{100*x['lo']:.0f},{100*x['hi']:.0f}]"
for t,dd in d['haiku'].items():
    if t=='vt_recall': print(t, {L:round(v['mean'],2) for L,v in dd.items()}); continue
    print(t, ' | '.join(f'{L}: {c(v)}' for L,v in dd.items()))
for k in ['haiku_know','haiku_tokens','haiku_cutoff_share','literal_errors','vt_errors','onehop_flags_needle','onehop_hedged','onehop_mentions_gold','lit_vs_hop','haiku_effective','deff_literal','deff_onehop','cost']:
    v=d[k]
    if isinstance(v,dict) and 'k' in v: print(k, c(v))
    elif k=='haiku_cutoff_share': print(k, {t:c(x) for t,x in v.items()})
    else: print(k, v)
print('depth')
for t,dd in d['depth'].items(): print(' ',t, ' | '.join(f'{b}: {c(v)}' for b,v in dd.items()))
print('thinking')
for k,v in d['thinking'].items(): print(' ',k,'off',c(v['off']),'on',c(v['on']),'diff %.2f [%.2f,%.2f]'%(v['diff'],*v['ci']), v['mcnemar'], 'out tok %.0f'%v['out_tokens'])
for tag in ('q05','q15'):
    o=d['open_'+tag]; print(tag,'know',c(o['know']))
    for t,dd in o['tasks'].items(): print('  ',t,' | '.join(f'{L}: {c(v)}' for L,v in dd.items()))
    print('  mem', {L:(m['prefix_tokens'],round(m['kv_mib']),round(m['peak_mib']),m['prefill_s']) for L,m in o['mem'].items()})
