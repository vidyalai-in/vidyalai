# Compare the listing's image with the pipeline's own output for the same prompt, seed and settings.
import numpy as np, torch, runpy
from PIL import Image
runpy.run_path("cfg_loop.py")
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
ref = pipe("a lighthouse on a rocky cliff at sunset", num_inference_steps=20, guidance_scale=7.5,
           generator=torch.Generator("cuda").manual_seed(11)).images[0]
ref.save("lighthouse_pipe.png")
a = np.asarray(Image.open("lighthouse.png"), np.float32); b = np.asarray(ref, np.float32)
mse = ((a - b) ** 2).mean()
print("max abs diff", np.abs(a - b).max(), "PSNR", 10 * np.log10(255**2 / mse) if mse else "identical")
