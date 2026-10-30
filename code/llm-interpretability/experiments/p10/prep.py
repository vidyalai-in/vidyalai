# Fixed evaluation sets for every checkpoint (CPU only): repeated random tokens, validation windows,
# and sequences sampled from the unigram and bigram models. Saved to runs/evalsets.npz.
import sys, numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import dynamics as D
allowed = D.frequent_tokens()
rep = D.repeated_tokens(200, 50, seed=1, allowed=allowed, device="cpu").numpy()
sel = D.repeated_tokens(100, 50, seed=3, allowed=allowed, device="cpu").numpy()   # head selection set
icl = D.val_windows(1000, 512, seed=0, device="cpu").numpy()
nat = D.val_windows(200, 256, seed=2, device="cpu").numpy()
ng = D.NGram(device="cpu")
uni = ng.sample(1, 64, 512, seed=4); bi = ng.sample(2, 64, 512, seed=5)
# entropy floors: the best possible loss on each kind of sampled sequence
H_uni = -ng.negH_uni
H_bi = float(-ng.negH_bi.numpy()[bi[:, :-1]].mean())
starts, lens = D.val_documents(511)
print("allowed", len(allowed), "val docs >= 511 tokens", len(starts), "H_uni", H_uni, "H_bi(samples)", H_bi)
np.savez("/home/amit/.cache/vidyalai-scratch/p10/runs/evalsets.npz", rep=rep, sel=sel, icl=icl, nat=nat,
         uni=uni, bi=bi, H_uni=H_uni, H_bi=H_bi, n_allowed=len(allowed))
