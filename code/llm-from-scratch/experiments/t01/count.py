V=32768
def params(d,L,nq,nkv,ff,hd=64):
    attn=d*nq*hd*2+2*d*nkv*hd
    blk=attn+3*d*ff+2*d
    return L*blk, V*d, L*blk+V*d+d
for name,cfg in {"a":(384,6,6,2,1024),"b":(576,8,9,3,1536),"b2":(576,10,9,3,1536),"b3":(512,10,8,2,1536),
 "c":(768,8,12,4,2048),"c2":(768,10,12,4,2048),"c3":(640,12,10,2,1792),"d":(768,16,12,4,2048),"d2":(768,12,12,4,2048),"e":(896,12,14,2,2560)}.items():
    ne,emb,tot=params(*cfg)
    print(name,cfg,f"nonemb {ne/1e6:.1f}M emb {emb/1e6:.1f}M total {tot/1e6:.1f}M")
