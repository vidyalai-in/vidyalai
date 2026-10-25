import os, sys, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
torch.cuda.set_per_process_memory_fraction(0.97)   # raise OOM instead of spilling into system memory (WSL)

def heldout_batch(B, T, start=0):
    """B rows of T+1 consecutive held-out tokens: inputs x and next-token targets y."""
    ids = np.load("/home/amit/.cache/vidyalai-scratch/t04/heldout_ids.npy")
    rows = ids[start:start + B * (T + 1)].astype(np.int64).reshape(B, T + 1)
    t = torch.from_numpy(rows).cuda()
    return t[:, :-1], t[:, 1:]
