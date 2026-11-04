import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import sd
t0=time.time(); pipe = sd.load_pipe(); print("load s", time.time()-t0)
for n in ["unet","text_encoder","vae","safety_checker"]:
    m=getattr(pipe,n); print(n, sum(p.numel() for p in m.parameters())/1e6, "M", next(m.parameters()).dtype)
print("attn procs", set(type(p).__name__ for p in pipe.unet.attn_processors.values()))
print("mem alloc GB", torch.cuda.memory_allocated()/1e9)
sd.set_sampler(pipe,"dpmpp2m")
t0=time.time()
im = pipe("a red bicycle leaning against a blue wooden door", num_inference_steps=20, guidance_scale=7.5, generator=torch.Generator("cuda").manual_seed(0)).images[0]
torch.cuda.synchronize(); print("first image s", time.time()-t0)
t0=time.time()
out = pipe("a red bicycle leaning against a blue wooden door", num_inference_steps=20, guidance_scale=7.5, generator=torch.Generator("cuda").manual_seed(0))
torch.cuda.synchronize(); print("second image s", time.time()-t0, out.nsfw_content_detected)
out.images[0].save("/home/amit/.cache/vidyalai-scratch/v06/runs/smoke.png")
print("peak GB", torch.cuda.max_memory_allocated()/1e9)
