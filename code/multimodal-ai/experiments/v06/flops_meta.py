# U-Net forward FLOPs vs spatial size, counted on the meta device (no weights, no GPU).
# 64 x 64 is SD 1.5's latent for a 512-pixel image; 512 x 512 is what the same network would face in pixel space.
import json, torch
from diffusers import UNet2DConditionModel
from torch.utils.flop_counter import FlopCounterMode
cfg = UNet2DConditionModel.load_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(cfg).half()
R = {}
for s in [32, 64, 128, 256, 512]:
    x = torch.randn(1, 4, s, s, device="meta", dtype=torch.float16)
    c = torch.randn(1, 77, 768, device="meta", dtype=torch.float16)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        unet(x, torch.tensor(500, device="meta"), encoder_hidden_states=c)
    tot = fc.get_total_flops()
    by = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
    attn = sum(v for k, v in by.items() if "attention" in k or "bmm" in k or "baddbmm" in k)
    R[s] = {"gflops": tot / 1e9, "attn_gflops": attn / 1e9, "by_op": {k: v / 1e9 for k, v in by.items()}}
    print(s, f"{tot/1e9:.0f} GFLOP", f"attention-core {attn/1e9:.0f}", flush=True)
json.dump(R, open("runs/flops_meta.json", "w"), indent=1)
