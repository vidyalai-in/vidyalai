import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                               dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
cond, uncond = pipe.encode_prompt("a lighthouse on a rocky cliff at sunset", "cuda", 1, True, negative_prompt="")

sched, s = pipe.scheduler, 7.5                    # s is the guidance scale
sched.set_timesteps(20)
g = torch.Generator("cuda").manual_seed(11)
lat = torch.randn((1, 4, 64, 64), generator=g, device="cuda", dtype=torch.float16) * sched.init_noise_sigma
with torch.no_grad():
    for t in sched.timesteps:                     # 20 steps, two U-Net passes in one batch each
        x = sched.scale_model_input(torch.cat([lat, lat]), t)
        eps_u, eps_c = pipe.unet(x, t, encoder_hidden_states=torch.cat([uncond, cond])).sample.chunk(2)
        lat = sched.step(eps_u + s * (eps_c - eps_u), t, lat).prev_sample
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
    img, flagged = pipe.run_safety_checker(img, "cuda", torch.float16)
pipe.image_processor.postprocess(img)[0].save("lighthouse.png")
print("flagged:", flagged)
