# Cosine similarity structure of ViT-B/16's learned 1D position embeddings (CPU, reads one tensor).
import glob, json, numpy as np, torch
from safetensors import safe_open
HF="/home/amit/.cache/vidyalai-scratch/hf/hub"
out={}
for m,g in [("google--vit-base-patch16-224",14),("google--vit-base-patch16-384",24)]:
    f=glob.glob(f"{HF}/models--{m}/snapshots/*/model.safetensors")[0]
    with safe_open(f,"pt") as s:
        key=[k for k in s.keys() if "position_embeddings" in k][0]
        pe=s.get_tensor(key).float()[0]          # (1+g*g, 768)
    p=torch.nn.functional.normalize(pe[1:],dim=-1); S=(p@p.T).numpy()
    ys,xs=np.divmod(np.arange(g*g),g)
    dist=np.sqrt((ys[:,None]-ys[None])**2+(xs[:,None]-xs[None])**2)
    bins={}
    for d in [1,2,3,4,6,8,10,13]:
        msk=(np.abs(dist-d)<0.5)&(dist>0)
        bins[d]=float(S[msk].mean())
    same_row=float(S[(ys[:,None]==ys[None])&(dist>0)].mean()); diff=float(S[(ys[:,None]!=ys[None])&(xs[:,None]!=xs[None])].mean())
    cls_sim=float(torch.nn.functional.cosine_similarity(pe[0:1],pe[1:]).mean())
    q=[(0,0),(g//2,g//2),(g//4, g-1-g//4)]
    maps={f"{r},{c}":S[r*g+c].reshape(g,g).round(3).tolist() for r,c in q}
    out[m]={"key":key,"grid":g,"shape":list(pe.shape),"mean_cos_by_dist":bins,"same_row_or_col_mean":same_row,
            "neither_mean":diff,"cls_vs_patch_mean":cls_sim,"maps":maps,"norm_mean":float(pe[1:].norm(dim=-1).mean())}
    print(m,pe.shape,{k:round(v,3) for k,v in bins.items()},round(same_row,3),round(diff,3),round(cls_sim,3))
# same-row-or-col vs neither, matched by distance (distance 3..6 band)
json.dump(out,open("/home/amit/.cache/vidyalai-scratch/v02/results/pos_embed.json","w"))
