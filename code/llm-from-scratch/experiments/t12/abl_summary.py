import json, statistics as st, sys
R='/home/amit/.cache/vidyalai-scratch/t12/runs/'
out={}
for d in sys.argv[1:]:
    rec=[json.loads(l) for l in open(R+d+'/log.jsonl')]
    tr=[r for r in rec if 'reward' in r][:100]
    te=[r for r in rec if 'test_acc' in r]
    f=lambda k,a,b: round(st.mean(r[k] for r in tr[a:b]),4)
    out[d]={'test':[(r['step'],r['test_acc'],r.get('test_acc_contains')) for r in te if r['step']<=100],
            'reward_90_99':f('reward',90,100),'kl_90_99':f('kl_tok',90,100),'ent_90_99':f('entropy',90,100),
            'len_90_99':f('len',90,100),'worked_90_99':f('worked',90,100),'zv_90_99':f('zero_var',90,100),
            'clip_mean':f('clip_frac',0,100),'gn_mean':f('grad_norm',0,100),'t_train':f('t_train',0,100),'t_gen':f('t_gen',0,100),
            'steps':len(tr), 'elapsed_100': tr[-1]['elapsed']}
    print(d, out[d])
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t12/post/ablations.json','w'), indent=1)
