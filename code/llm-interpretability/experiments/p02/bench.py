import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import lens as LL
m = LL.load("gpt2")
d = torch.load("/home/amit/.cache/vidyalai-scratch/p02/runs/tokens-gpt2.pt")
tok = d["train"][:8].cuda()

def t(f, n=5):
    f(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1000

logits, H = LL.resid_points(m, tok)
h = H[5][:, 1:].clone().requires_grad_(True)
flp = F.log_softmax(logits[:, 1:].float(), -1); p = flp.exp()
print("forward+cache ms", t(lambda: LL.resid_points(m, tok)))
def a():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = LL.logit_lens(m, h)
    return z
print("unembed bf16 fwd ms", t(a))
def b():
    z = a(); loss = LL.kl_from_final(flp, z).mean(); loss.backward()
print("old kl fwd+bwd ms", t(b))
def c():
    z = a(); loss = F.cross_entropy(z.flatten(0, 1).float(), p.flatten(0, 1)); loss.backward()
print("soft-CE fwd+bwd ms", t(c))
def e():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = LL.logit_lens(m, h)
        loss = F.cross_entropy(z.flatten(0, 1), p.flatten(0, 1))
    loss.backward()
print("soft-CE in autocast ms", t(e))
torch.backends.cuda.matmul.allow_tf32 = True
print("tf32 fp32 unembed fwd ms", t(lambda: LL.logit_lens(m, h.detach())))
torch.backends.cuda.matmul.allow_tf32 = False
print("fp32 unembed fwd ms", t(lambda: LL.logit_lens(m, h.detach())))
print("peak GiB", torch.cuda.max_memory_allocated() / 2**30)
