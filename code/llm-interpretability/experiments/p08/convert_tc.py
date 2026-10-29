# Convert the 26 downloaded Gemma Scope transcoder .npz files to bf16 safetensors (lazy-loadable).
import sys, os, glob
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.attribution import convert_gemma_scope_npz, TC_DIR
for f in sorted(glob.glob(os.path.join(TC_DIR, "*.npz"))):
    out = f[:-4] + ".safetensors"
    if not os.path.exists(out):
        print(os.path.basename(out), convert_gemma_scope_npz(f, out), flush=True)
