import sys, time; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import torch, ar_image as ai
torch.set_float32_matmul_precision("high")
for N in (49, 196):
    codes = torch.randint(0, 512, (60000, N))
    for mode in ("ar", "mask"):
        for d, L in ((256, 6), (192, 6)):
            for bs in (128, 256):
                m = ai.Prior(512, N, mode, d=d, L=L).cuda()
                ai.train_prior(codes, 512, mode, steps=20, bs=bs, model=m)
                torch.cuda.synchronize(); t0 = time.time()
                ai.train_prior(codes, 512, mode, steps=100, bs=bs, model=m)
                torch.cuda.synchronize()
                print(N, mode, d, L, bs, "ms/step", round((time.time() - t0) * 10, 1), "params", sum(p.numel() for p in m.parameters()), flush=True)
