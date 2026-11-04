"""Save 64 real training images and their VQ-VAE reconstructions for the sample figure (CPU is fine)."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import torch, fmnist_eval as fe, ar_image as ai
cfg = json.load(open("/home/amit/.cache/vidyalai-scratch/v08/prior_cfg.json"))
m = ai.VQVAE(K=cfg["K"], down=cfg["down"]); m.load_state_dict(torch.load(cfg["vq"], map_location="cpu")); m.eval()
x, _ = fe.load("train")
idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(11))[:64]
with torch.no_grad():
    r = m.decode(m.encode(x[idx])).clamp(-1, 1)
R = "/home/amit/.cache/vidyalai-scratch/v08/runs/"
torch.save(x[idx].half(), R + "grid_real.pt"); torch.save(r.half(), R + "grid_recon.pt"); print("ok")
