import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

img = Image.open("scene.png").convert("RGB")                    # 512 x 512
proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTModel.from_pretrained("google/vit-base-patch16-224", dtype=torch.bfloat16, device_map="cuda")
x = proc(images=img, return_tensors="pt")["pixel_values"]       # (1, 3, 224, 224)
with torch.no_grad():
    print(vit(pixel_values=x.to("cuda", torch.bfloat16)).last_hidden_state.shape)
# torch.Size([1, 197, 768])

import numpy as np
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.bfloat16).to("cuda")
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].div(127.5).sub(1)   # (1, 3, 512, 512) in [-1, 1]
with torch.no_grad():
    z = vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mean
print(z.shape)                           # torch.Size([1, 4, 64, 64])
