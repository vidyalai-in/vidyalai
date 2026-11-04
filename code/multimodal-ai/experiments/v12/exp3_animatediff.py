"""Experiment 3: AnimateDiff (motion module mm_sd_v15_v2, Apache-2.0, guoyww/animatediff) on Stable Diffusion 1.5
(stable-diffusion-v1-5/stable-diffusion-v1-5, CreativeML OpenRAIL-M), fp16 on the RTX 4060 Laptop GPU.
--phase smoke   : load, one 16-frame clip at 4 steps, peak VRAM
--phase timing  : per-step U-Net time (median of 5 after 2 warm-up) and peak VRAM vs number of frames
--phase clips   : 25-step clips for the prompts below, safety checker on every frame, CLIP frame-to-frame similarity,
                  plus the same prompt and seed through plain SD 1.5 (no motion module) for comparison
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import video as V
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor

ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True)
ap.add_argument("--frames", default="1,4,8,16,24,32"); ap.add_argument("--size", type=int, default=512)
ap.add_argument("--prompts", default="0,1,2,3"); ap.add_argument("--nm_prompts", default="0,1")
a = ap.parse_args()
phases = a.phase.split(",")
# Raise OutOfMemoryError at the card's size instead of letting the WSL driver spill into system RAM.
torch.cuda.set_per_process_memory_fraction(0.95)
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPTS = ["a sailboat on a calm lake at sunset, gentle waves",
           "a red fox walking through fresh snow in a pine forest",
           "a campfire burning at night, sparks rising",
           "ocean waves crashing on a rocky shore",
           "a hot air balloon floating over green hills",
           "a cup of coffee with steam rising on a wooden table"]
NEG = "low quality, blurry"
OUT = Path("runs"); FR = Path("frames"); FR.mkdir(exist_ok=True)

t0 = time.time()
from diffusers import UNet2DConditionModel, UNetMotionModel
ckpt = hf_hub_download("guoyww/animatediff", "mm_sd_v15_v2.ckpt")
adapter = MotionAdapter.from_single_file(ckpt, torch_dtype=torch.float16).to("cuda")
unet2d = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to("cuda")
torch.set_default_dtype(torch.float16)
with torch.device("cuda"):   # from_unet2d builds a fresh model: on the CPU in fp32 that is 5 GB of host RAM
    unet = UNetMotionModel.from_unet2d(unet2d, adapter)
torch.set_default_dtype(torch.float32)
del unet2d; torch.cuda.empty_cache()
pipe = AnimateDiffPipeline.from_pretrained(SD, unet=unet, motion_adapter=adapter, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, timestep_spacing="linspace",
                                           beta_schedule="linear", steps_offset=1)
pipe.set_progress_bar_config(disable=True)
n_mm = sum(p.numel() for n, p in pipe.unet.named_parameters() if "motion_modules" in n)
n_unet = sum(p.numel() for p in pipe.unet.parameters())
print(f"loaded in {time.time()-t0:.0f}s; unet params {n_unet/1e6:.1f}M of which motion modules {n_mm/1e6:.1f}M; "
      f"weights on GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
info = {"unet_params": n_unet, "motion_params": n_mm, "weights_gib": torch.cuda.memory_allocated() / 2**30}

def run(prompt, frames, steps, seed=42, size=a.size, ptype="pil"):
    return pipe(prompt, negative_prompt=NEG, num_frames=frames, height=size, width=size, num_inference_steps=steps,
                guidance_scale=7.5, generator=torch.Generator("cpu").manual_seed(seed), output_type=ptype, decode_chunk_size=4).frames[0]

if "smoke" in phases:
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    fr = run(PROMPTS[0], 16, 4)
    torch.cuda.synchronize()
    print("smoke 16 frames 4 steps", time.time() - t, "s peak", torch.cuda.max_memory_allocated() / 2**30, "GiB", flush=True)

if "timing" in phases:
    rows = []
    unet = pipe.unet
    emb = pipe.encode_prompt(PROMPTS[0], "cuda", 1, True, negative_prompt=NEG)
    ctx = torch.cat([emb[1], emb[0]])
    for F_ in [int(f) for f in a.frames.split(",")]:
        lat = torch.randn(2, 4, F_, a.size // 8, a.size // 8, device="cuda", dtype=torch.float16)  # CFG batch of 2
        t = torch.tensor(500, device="cuda")
        ctx_f = ctx.repeat_interleave(F_, dim=0)  # the pipeline repeats the text states once per frame
        def step():
            with torch.no_grad():
                unet(lat, t, encoder_hidden_states=ctx_f).sample
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            with V.GpuMonitor() as mon:
                med, lo, hi = V.cuda_median(step, n=5, warmup=2)
            r = {"frames": F_, "size": a.size, "step_median_s": med, "step_min_s": lo, "step_max_s": hi,
                 "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "smi_mem_max_mib": mon.mem_max,
                 "sm_clock_mhz": mon.clock, "power_w": mon.power, "n": 5}
        except torch.OutOfMemoryError:
            r = {"frames": F_, "size": a.size, "oom": True}; torch.cuda.empty_cache()
        rows.append(r); print(json.dumps(r), flush=True)
    json.dump({"rows": rows, **info}, open(OUT / f"exp3_timing_{a.size}.json", "w"), indent=1)

if "clips" in phases:
    sc = StableDiffusionSafetyChecker.from_pretrained(SD, subfolder="safety_checker", torch_dtype=torch.float16).to("cuda")
    fx = CLIPImageProcessor.from_pretrained(SD, subfolder="feature_extractor")
    def check(imgs):
        inp = fx(imgs, return_tensors="pt").pixel_values.to("cuda", torch.float16)
        arr = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in imgs])
        _, flags = sc(images=arr, clip_input=inp)
        return [bool(f) for f in flags]
    res = []
    for i in [int(p) for p in a.prompts.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        with V.GpuMonitor() as mon:
            torch.cuda.synchronize(); t = time.time()
            fr = run(PROMPTS[i], 16, 25)
            torch.cuda.synchronize(); dt = time.time() - t
        flags = check(fr)
        d = FR / f"ad-p{i}"; d.mkdir(exist_ok=True)
        for k, im in enumerate(fr): im.save(d / f"{k:02d}.png")
        r = {"prompt_id": i, "prompt": PROMPTS[i], "frames": 16, "size": a.size, "steps": 25, "seed": 42, "guidance": 7.5,
             "sampler": "DDIM (linear betas, linspace spacing, clip_sample off)", "negative": NEG,
             "clip_s": dt, "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "sm_clock_mhz": mon.clock,
             "power_w": mon.power, "flags": flags}
        res.append(r); print(json.dumps(r), flush=True)
    json.dump({"clips": res, **info}, open(OUT / "exp3_clips.json", "w"), indent=1)

if "nomotion" in phases:
    # The motion modules are residual: output = input + temporal transformer(input). Returning the input
    # switches them off and leaves plain Stable Diffusion 1.5, denoising each frame from its own noise.
    from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D
    AnimateDiffTransformer3D.forward = lambda self, hidden_states, *args, **kw: hidden_states
    for i in [int(p) for p in a.nm_prompts.split(",")]:
        torch.cuda.synchronize(); t = time.time()
        fr = run(PROMPTS[i], 16, 25)
        torch.cuda.synchronize(); dt = time.time() - t
        d = FR / f"nomotion-p{i}"; d.mkdir(exist_ok=True)
        for k, im in enumerate(fr): im.save(d / f"{k:02d}.png")
        print(json.dumps({"prompt_id": i, "nomotion_clip_s": dt}), flush=True)
