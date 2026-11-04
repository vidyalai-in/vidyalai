# Experiment 3: what the latent looks like during sampling. DDIM, 20 steps, guidance 7.5, one prompt and seed.
# Saves the decoded noisy latent x_t and the decoded prediction of the clean latent x0 at chosen steps,
# plus the raw 4-channel latents, and runs the safety checker on every decoded image.
import json, sys, torch, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import sd
import os
os.makedirs("runs/x0", exist_ok=True)
pipe = sd.load_pipe()
sd.set_sampler(pipe, "ddim")
R = []
for prompt, seed in [("a lighthouse on a rocky cliff at sunset", 11), ("a red bicycle leaning against a blue wooden door", 22)]:
    cond, uncond = sd.encode(pipe, prompt)
    sched = pipe.scheduler
    sched.set_timesteps(20, device="cuda")
    g = torch.Generator("cuda").manual_seed(seed)
    lat = torch.randn((1, 4, 64, 64), generator=g, device="cuda", dtype=torch.float16) * sched.init_noise_sigma
    ctx = torch.cat([uncond, cond])
    tag = prompt.split()[1]
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            eps_u, eps_c = pipe.unet(torch.cat([lat, lat]), t, encoder_hidden_states=ctx).sample.chunk(2)
            eps = eps_u + 7.5 * (eps_c - eps_u)
            out = sched.step(eps, t, lat)
            x0 = out.pred_original_sample
            xt_img = sd.decode(pipe, lat)[0]
            x0_img = sd.decode(pipe, x0)[0]
            flags, margins = sd.safety(pipe, [xt_img, x0_img])
            xt_img.save(f"runs/x0/{tag}_xt_{i:02d}.png"); x0_img.save(f"runs/x0/{tag}_x0_{i:02d}.png")
            np.save(f"runs/x0/{tag}_lat_{i:02d}.npy", lat[0].float().cpu().numpy())
            np.save(f"runs/x0/{tag}_x0lat_{i:02d}.npy", x0[0].float().cpu().numpy())
            a = sched.alphas_cumprod[int(t)].item()
            R.append(dict(prompt=prompt, seed=seed, step=i, t=int(t), alpha_bar=a, snr=a / (1 - a),
                          lat_std=float(lat.float().std()), nsfw=flags, margins=margins))
            lat = out.prev_sample
        final = sd.decode(pipe, lat)[0]; final.save(f"runs/x0/{tag}_final.png")
        f, m = sd.safety(pipe, [final]); R.append(dict(prompt=prompt, final=True, nsfw=f, margins=m))
json.dump(R, open("runs/exp3_x0.json", "w"), indent=1)
print("flags", sum(sum(r["nsfw"]) for r in R))
