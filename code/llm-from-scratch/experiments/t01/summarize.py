import json, datetime, statistics as st
D='/home/amit/.cache/vidyalai-scratch/t01/logs/'
out={}
for s in ['t20','t50','t85','t125']:
    steps=[];res=None
    for l in open(D+f'full_{s}.out'):
        if l.startswith('{"step"'): steps.append(json.loads(l))
        if l.startswith('RESULT'): res=json.loads(l[7:])
    t0=steps[0]['t']-steps[0]['s']; t1=steps[-1]['t']
    rows=[]
    for l in open(D+f'full_{s}.smi.csv'):
        f=[x.strip() for x in l.split(',')]
        ts=datetime.datetime.strptime(f[0],'%Y/%m/%d %H:%M:%S.%f').timestamp()
        if t0<=ts<=t1: rows.append((int(f[1].split()[0]),int(f[2].split()[0]),float(f[3].split()[0]),float(f[5].split()[0]),int(f[6]),f[8]))
    sm=[r[0] for r in rows]; mem=[r[1] for r in rows]; pw=[r[2] for r in rows]; lim=[r[3] for r in rows]; tmp=[r[4] for r in rows]
    tps=[x['tok_s'] for x in steps]
    fpt=res['flops_per_token']; N=res['N']; d=res['d']; L=res['L']
    med=res['tok_s']
    r=dict(size=s,N=N,nonemb=res['nonemb'],mb=res['mb'],accum=res['accum'],steps=len(steps),window_s=round(t1-t0),
      step_s_med=round(res['step_s_median'],2),tok_s=round(med),tok_s_min=min(tps),tok_s_max=max(tps),
      tflops=round(med*fpt/1e12,2),mfu_palm=round(med*fpt/25.3e12*100,1),mfu_6N=round(med*6*N/25.3e12*100,1),
      mfu_kaplan=round(med*(6*N+6*L*d*1024)/25.3e12*100,1),
      peak_GiB=round(res['peak_mem_GiB'],2),reserved_GiB=round(res['reserved_GiB'],2),
      sm=(min(sm),round(st.median(sm)),max(sm)),memclk=(min(mem),max(mem)),power=(round(min(pw),1),round(st.mean(pw),1),round(max(pw),1)),
      limit=(min(lim),max(lim)),temp=(min(tmp),max(tmp)),reasons=sorted(set(r[5] for r in rows)),loss_last=steps[-1]['loss'],warm=res['warmup_s'])
    out[s]=r; print(json.dumps(r))
json.dump(out,open(D+'summary.json','w'),indent=1)
