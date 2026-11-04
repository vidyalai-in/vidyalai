import torch, time, torch.nn.functional as F
B,T,S,h,hd = 32,8,144,6,32
x = torch.randn(B,T,S,3*h*hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
def sdpa_t():
    q,k,v = x.reshape(B,T,S,3,h,hd).permute(3,0,2,4,1,5)  # (B,S,h,T,hd)
    q,k,v = [t.reshape(B*S,h,T,hd) for t in (q,k,v)]
    o = F.scaled_dot_product_attention(q,k,v)
    return o.reshape(B,S,h,T,hd).permute(0,3,1,2,4).reshape(B,T,S,h*hd)
def ein_t():
    q,k,v = x.reshape(B,T,S,3,h,hd).unbind(3)
    s = torch.einsum('btshd,bushd->bshtu', q, k) * hd**-0.5
    p = s.float().softmax(-1).to(q.dtype)
    return torch.einsum('bshtu,bushd->btshd', p, v).reshape(B,T,S,h*hd)
def sdpa_s():
    q,k,v = x.reshape(B*T,S,3,h,hd).permute(2,0,3,1,4)
    return F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,T,S,h*hd)
print((sdpa_t()-ein_t()).abs().max().item())
for fn in (sdpa_t, ein_t, sdpa_s):
    for _ in range(3): fn().sum().backward()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): fn().sum().backward()
    torch.cuda.synchronize(); print(fn.__name__, (time.time()-t0)/20*1000, "ms fwd+bwd")
