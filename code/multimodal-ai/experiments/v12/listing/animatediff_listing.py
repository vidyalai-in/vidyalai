import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel
from huggingface_hub import hf_hub_download

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ckpt = hf_hub_download("guoyww/animatediff", "mm_sd_v15_v2.ckpt")           # motion module, Apache-2.0
adapter = MotionAdapter.from_single_file(ckpt, torch_dtype=torch.float16).to("cuda")
unet2d = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", variant="fp16",
                                              torch_dtype=torch.float16).to("cuda")
torch.set_default_dtype(torch.float16)
with torch.device("cuda"):                       # build the video U-Net on the GPU, in fp16, not in host RAM
    unet = UNetMotionModel.from_unet2d(unet2d, adapter)   # SD 1.5's layers plus the temporal motion modules
torch.set_default_dtype(torch.float32)
del unet2d
pipe = AnimateDiffPipeline.from_pretrained(SD, unet=unet, motion_adapter=adapter, variant="fp16",
                                           torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, timestep_spacing="linspace",
                                           beta_schedule="linear", steps_offset=1)
frames = pipe("a sailboat on a calm lake at sunset, gentle waves", negative_prompt="low quality, blurry",
              num_frames=16, height=512, width=512, num_inference_steps=25, guidance_scale=7.5,
              decode_chunk_size=4, generator=torch.Generator("cpu").manual_seed(42)).frames[0]  # 16 PIL images
print(len(frames), frames[0].size)
