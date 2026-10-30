# Per-checkpoint measurements for one run: head scores, in-context learning, n-gram KL and n-gram
# sequence loss. Usage: track.py RUN [max_seconds]. Resumable: skips checkpoints already done.
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import dynamics as D
run = sys.argv[1]; budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
OUT = f"/home/amit/.cache/vidyalai-scratch/p10/runs/track/{run}"; os.makedirs(OUT, exist_ok=True)
E = np.load("/home/amit/.cache/vidyalai-scratch/p10/runs/evalsets.npz")
g = lambda k: torch.as_tensor(E[k], device="cuda")
rep, icl, nat, uni, bi = g("rep"), g("icl"), g("nat"), g("uni"), g("bi")
ng = D.NGram()
t0 = time.time()
for step, path in D.snapshots(run):
    f = f"{OUT}/{step:05d}.npz"
    if os.path.exists(f): continue
    if time.time() - t0 > budget: print("budget reached"); break
    t1 = time.time()
    m = D.load(path, D.RUNS[run].cfg)
    r = D.head_scores(m, rep, half=50)                    # FP32, patterns recomputed
    n = D.head_scores(m, nat, half=None)
    L = D.token_losses(m, icl)                            # BF16 autocast
    ku, kb = ng.kl(m, icl[:64])
    lu = D.token_losses(m, uni); lb = D.token_losses(m, bi)
    np.savez(f, step=step, tokens=step * D.RUNS[run].tokens_per_step,
             rep_ind=r["induction"].astype(np.float32), rep_dup=r["dup"].astype(np.float32),
             rep_prev=r["prev"].astype(np.float32), rep_loss=r["loss"].mean(0),
             rep_first=r["loss"][:, 1:50].mean(1), rep_second=r["loss"][:, 51:].mean(1),
             nat_prev=n["prev"].astype(np.float32),
             icl_curve=L.mean(0), icl_L50=L[:, 49], icl_L500=L[:, 499],
             icl_s50=L[:, 40:60].mean(1), icl_s500=L[:, 490:510].mean(1), icl_mean=L.mean(1),
             kl_uni=np.nanmean(ku, 1), kl_bi=np.nanmean(kb, 1), kl_uni_curve=np.nanmean(ku, 0), kl_bi_curve=np.nanmean(kb, 0),
             uni_loss=lu.mean(1), bi_loss=lb.mean(1), uni_curve=lu.mean(0), bi_curve=lb.mean(0))
    ind = r["induction"].mean(0)
    print(f"{run} step {step}: ind max {ind.max():.3f} 2nd-copy {r['loss'][:, 51:].mean():.2f} "
          f"ICL {(L[:, 499]-L[:, 49]).mean():+.3f} loss {L.mean():.3f} KLu {np.nanmean(ku):.2f} KLb {np.nanmean(kb):.2f} "
          f"uni {lu.mean():.2f} bi {lb.mean():.2f} | {time.time()-t1:.1f}s", flush=True)
    del m; torch.cuda.empty_cache()
print("done", time.time() - t0, "peak GB", torch.cuda.max_memory_allocated() / 1e9)
