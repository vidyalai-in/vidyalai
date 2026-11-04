import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V, fmnist_eval as fe
torch.backends.cuda.matmul.allow_tf32 = True
x, y = fe.load("train"); x = x.cuda()
for mode in ["frames", "factorized", "full"]:
    for bs in [32, 64]:
        net = V.VideoDiT(mode).cuda(); opt = torch.optim.AdamW(net.parameters(), 3e-4)
        def step():
            d = V.moving_fmnist(x, bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = V.flow_loss(net, d["x"])
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5): step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(30): step()
        torch.cuda.synchronize(); dt = (time.time() - t0) / 30
        print(mode, bs, f"{dt*1000:.1f} ms/step  {bs/dt:.0f} clips/s  peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
        del net, opt; torch.cuda.empty_cache()
