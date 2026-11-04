"""Save reconstructions of the test picture through the SD VAE (continuous latent) and the aMUSEd VQ-GAN
(discrete codes), plus the latent itself, for the shown figure. GPU, short."""
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
from PIL import Image
import tokens

OUT = "/home/amit/.cache/vidyalai-scratch/v01/runs"
img = tokens.scene(512, 512)
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().div(127.5).sub(1)


def to_img(t):
    a = ((t[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


with torch.no_grad():
    from diffusers import AutoencoderKL, VQModel
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.bfloat16).to("cuda").eval()
    z = vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mean
    to_img(vae.decode(z).sample).save(f"{OUT}/recon_vae.png")
    np.save(f"{OUT}/latent_vae.npy", z[0].float().cpu().numpy())  # 4 x 64 x 64, 64 KB
    del vae
    torch.cuda.empty_cache()
    vq = VQModel.from_pretrained("amused/amused-256", subfolder="vqvae", variant="fp16", torch_dtype=torch.bfloat16).to("cuda").eval()
    h = vq.encode(x.to("cuda", torch.bfloat16)).latents
    q, _, info = vq.quantize(h)
    np.save(f"{OUT}/codes_vq.npy", info[2].reshape(h.shape[-2], h.shape[-1]).cpu().numpy().astype(np.int16))
    to_img(vq.decode(info[2].reshape(1, *h.shape[-2:]), force_not_quantize=True, shape=(1, *h.shape[-2:], h.shape[1])).sample).save(f"{OUT}/recon_vq.png")
img.save(f"{OUT}/orig_512.png")
print("ok")
