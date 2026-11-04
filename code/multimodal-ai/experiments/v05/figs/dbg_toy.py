import sys; sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/mm/lab'); sys.path.insert(0,'/home/amit/.cache/vidyalai-scratch/v05'); sys.argv=['x']
import torch, ddpm, toy, time, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(4)
steps=int(sys.argv[1]) if len(sys.argv)>1 else 20000
import builtins
torch.manual_seed(0); g=torch.Generator().manual_seed(0)
s=ddpm.Schedule('linear'); m=ddpm.MLPDenoiser()
opt=torch.optim.AdamW(m.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20000)
data=toy.spiral(100000,g); t0=time.time(); outs=[]
for it in range(20000):
    x0=data[torch.randint(0,len(data),(1024,),generator=g)]
    loss=ddpm.loss_simple(m,x0,s); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it+1 in (6000,20000):
        m.eval(); x=ddpm.sample(m,(2000,2),s,steps=100,method='ddim',device='cpu',clip=False,generator=torch.Generator().manual_seed(0)); outs.append(x); m.train(); print(it+1, time.time()-t0, flush=True)
f,ax=plt.subplots(1,2,figsize=(8,4)); [a.scatter(*x.T,s=2) for a,x in zip(ax,outs)]; plt.savefig('figs/dbg2.png',dpi=70)
