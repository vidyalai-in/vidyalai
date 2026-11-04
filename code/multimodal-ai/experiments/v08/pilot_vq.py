import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
torch_dummy=0
import torch, fmnist_eval as fe, ar_image as ai
torch.set_float32_matmul_precision("high")
x, _ = fe.load("train"); xt, _ = fe.load("test")
out = {}
for down in (4, 2):
    t0 = time.time()
    m, log = ai.train_vqvae(x, K=512, down=down, steps=4000, lr=1e-3, restart_every=200)
    torch.cuda.synchronize(); dt = time.time() - t0
    ev, _ = ai.vq_eval(m, xt)
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:5000]
    _, rec = ai.vq_eval(m, x[idx])
    fd = fe.score(rec)["fd"]
    out[down] = {"sec": dt, **ev, "fd_recon5000": fd, "log": log[-1]}
    print(down, json.dumps(out[down]), flush=True)
