import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V, fmnist_eval as fe
x, y = fe.load("train"); x = x.cuda()
for mode in ["frames", "factorized", "full"]:
    bs = 32
    net = V.VideoDiT(mode).cuda(); opt = torch.optim.AdamW(net.parameters(), 3e-4, fused=True)
    cnet = torch.compile(net)
    def step():
        d = V.moving_fmnist(x, bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = V.flow_loss(cnet, d["x"])
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.time()
    for _ in range(3): step()
    torch.cuda.synchronize(); print("compile+3", time.time() - t0, flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(30): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 30
    print(mode, bs, f"{dt*1000:.1f} ms/step  {bs/dt:.0f} clips/s  peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
    del net, opt, cnet; torch.cuda.empty_cache()
