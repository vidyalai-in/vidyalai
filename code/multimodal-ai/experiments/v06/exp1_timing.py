# Experiment 1: where the time and memory go in one SD 1.5 image at 512 x 512 (fp16, RTX 4060 Laptop GPU).
import json, sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import sd
from torch.utils.flop_counter import FlopCounterMode
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

R = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
t0 = time.perf_counter(); pipe = sd.load_pipe(); torch.cuda.synchronize()
R["load_s"] = time.perf_counter() - t0
R["params_M"] = {n: sum(p.numel() for p in getattr(pipe, n).parameters()) / 1e6
                 for n in ["unet", "text_encoder", "vae", "safety_checker"]}
R["weights_GB"] = torch.cuda.memory_allocated() / 1e9
unet, te, vae = pipe.unet, pipe.text_encoder, pipe.vae
prompt = "a lighthouse on a rocky cliff at sunset"
ids = pipe.tokenizer([prompt, ""], padding="max_length", max_length=77, return_tensors="pt").input_ids.cuda()
g = torch.Generator("cuda").manual_seed(0)
lat = torch.randn((16, 4, 64, 64), generator=g, device="cuda", dtype=torch.float16)
with torch.no_grad():
    ctx2 = te(ids)[0]
ctx1 = ctx2[:1]
t = torch.tensor(500, device="cuda")


def blk(name, fn, n=20, warmup=5):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), sd.GpuMonitor() as m:
        med, lo, hi = sd.cuda_median(fn, n, warmup)
    R.setdefault("timings", {})[name] = {"median_ms": med * 1e3, "min_ms": lo * 1e3, "max_ms": hi * 1e3, "n": n,
                                         "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                                         "clock_MHz": m.clock, "power_W": m.power, "temp_C": m.temp}
    print(name, f"{med*1e3:.1f} ms", f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", m.clock, m.power, flush=True)


blk("text_encoder_b2", lambda: te(ids))
for mode in ["sdpa", "classic", "sliced"]:
    if mode == "sdpa":
        unet.set_attn_processor(AttnProcessor2_0())
    elif mode == "classic":
        unet.set_attn_processor(AttnProcessor())
    else:
        unet.set_attn_processor(AttnProcessor2_0()); pipe.enable_attention_slicing()
    blk(f"unet_{mode}_b1", lambda: unet(lat[:1], t, encoder_hidden_states=ctx1))
    blk(f"unet_{mode}_b2", lambda: unet(lat[:2], t, encoder_hidden_states=ctx2))
    blk(f"unet_{mode}_2x1", lambda: (unet(lat[:1], t, encoder_hidden_states=ctx1),
                                     unet(lat[1:2], t, encoder_hidden_states=ctx2[1:])))
    pipe.disable_attention_slicing() if mode == "sliced" else None
unet.set_attn_processor(AttnProcessor2_0())
for b in [4, 8, 16]:
    blk(f"unet_sdpa_b{b}", lambda: unet(lat[:b], t, encoder_hidden_states=ctx2.repeat(b // 2, 1, 1)), n=10, warmup=3)
z = lat[:1] / sd.VAE_SCALE
blk("vae_decode_b1", lambda: vae.decode(z))
vae.set_attn_processor(AttnProcessor())
blk("vae_decode_b1_classic", lambda: vae.decode(z))
vae.set_attn_processor(AttnProcessor2_0())
img = sd.decode(pipe, lat[:1])
blk("safety_checker", lambda: sd.safety(pipe, img))

# FLOPs (forward) for each part
def flops(fn):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()
R["gflops"] = {"text_encoder_b1": flops(lambda: te(ids[:1])) / 1e9,
               "unet_b1": flops(lambda: unet(lat[:1], t, encoder_hidden_states=ctx1)) / 1e9,
               "vae_decode_b1": flops(lambda: vae.decode(z)) / 1e9}
print(R["gflops"], flush=True)

# End to end through the pipeline (includes text encoding, scheduler, VAE decode, safety checker, PIL conversion)
def e2e(sampler, steps, gs, n=5, warmup=1):
    sd.set_sampler(pipe, sampler)
    run = lambda: pipe(prompt, num_inference_steps=steps, guidance_scale=gs,
                       generator=torch.Generator("cuda").manual_seed(1))
    blk(f"e2e_{sampler}_{steps}_g{gs}", run, n=n, warmup=warmup)
e2e("dpmpp2m", 20, 7.5)
e2e("dpmpp2m", 20, 1.0)
e2e("pndm", 50, 7.5)
e2e("ddim", 50, 7.5)
json.dump(R, open("runs/exp1_timing.json", "w"), indent=1)
print("done")
