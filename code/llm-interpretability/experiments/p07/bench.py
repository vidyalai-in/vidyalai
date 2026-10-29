import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import sae as S
torch.backends.cuda.matmul.allow_tf32 = True
x = torch.randn(4096, 768, device="cuda") * 3.7
for kind, kw in [("topk", {"k": 32}), ("relu", {"l1": 5.0})]:
    for ac in [torch.bfloat16, None]:
        s = S.SAE(768, 24576, kind=kind, **kw).cuda(); s.scale.fill_(3.7)
        opt = torch.optim.Adam(s.parameters(), lr=1e-4)
        def step():
            if ac:
                with torch.autocast("cuda", dtype=ac):
                    loss, _ = S.loss_fn(s, x)
            else:
                loss, _ = S.loss_fn(s, x)
            opt.zero_grad(); loss.backward(); opt.step()
        for _ in range(5): step()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): step()
        torch.cuda.synchronize()
        print(kind, ac, f"{(time.time()-t)/20*1000:.1f} ms/step", flush=True)
# matmul roofline check
a = torch.randn(4096, 768, device="cuda", dtype=torch.bfloat16); b = torch.randn(768, 24576, device="cuda", dtype=torch.bfloat16)
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(50): a @ b
torch.cuda.synchronize(); dt = (time.time() - t) / 50
print(f"bf16 matmul 4096x768x24576: {dt*1000:.2f} ms = {2*4096*768*24576/dt/1e12:.1f} TFLOPS")
