"""Experiment 1c: how the masked prior chooses which codes to keep. choice_temp scales the Gumbel noise added to the
log-probability ranking (annealed to 0 over the passes); 0 = pure confidence order, 1e4 = effectively random order."""
import sys, json
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/v08"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import torch, fmnist_eval as fe, ar_image as ai
import prior_run as pr
vq = pr.load_vq(); mk, N = pr.load_prior("mask"); side = int(N ** 0.5)
out = pr.RUNS / "prior_fd.jsonl"
for T in (8, 49):
    for ct in (0.0, 1.0, 1e4):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            c = pr.gen(lambda b: ai.mask_sample(mk, b, T=T, choice_temp=ct))
            r = fe.score(pr.decode(vq, c, side)); r.update({"name": f"mask_T{T}_ct{ct:g}", "seed": seed})
            open(out, "a").write(json.dumps(r) + "\n"); print(r["name"], seed, round(r["fd"], 2), flush=True)
