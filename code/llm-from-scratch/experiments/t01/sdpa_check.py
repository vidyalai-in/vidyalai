import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
q=torch.randn(4,12,1024,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(4,4,1024,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
v=torch.randn_like(k,requires_grad=True)
for b in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
    try:
        with sdpa_kernel(b):
            F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True).sum().backward()
        print(b,"ok")
    except Exception as e: print(b,"FAIL",str(e)[:150])
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True).sum().backward()
print([e.key for e in p.key_averages() if 'flash' in e.key.lower() or 'attention' in e.key.lower() or 'fmha' in e.key.lower()][:6])
print(torch.__version__)
