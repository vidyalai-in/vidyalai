import math
V=32768
def cnt(d,L,hd):
    nh=d//hd; nkv=nh//3
    ff=math.ceil(8*d/3/256)*256
    attn=d*nh*hd+2*d*nkv*hd+nh*hd*d
    per=attn+3*d*ff+2*d
    ne=L*per+d
    return ne, V*d, ff, nh, nkv
rows=[]
for d in range(192,769,48):
    for hd in (48,64,80,96):
        if d%hd: continue
        nh=d//hd
        if nh%3: continue
        for L in range(2,17):
            ne,emb,ff,nh,nkv=cnt(d,L,hd)
            tot=ne+emb
            if 9e6<tot<50e6 and 24<=d/L<=160:
                rows.append((tot,ne,d,L,hd,nh,nkv,ff,d/L, ff/d))
rows.sort()
for r in rows: print("%6.2fM ne %6.2fM d=%d L=%d hd=%d h=%d/%d ff=%d aspect=%.0f ffr=%.2f"%(r[0]/1e6,r[1]/1e6,*r[2:]))
