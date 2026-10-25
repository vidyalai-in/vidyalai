import sys, time
from huggingface_hub import hf_hub_download
out = "/home/amit/.cache/vidyalai-scratch/fromscratch/data/fineweb-edu-10BT"
for i in range(int(sys.argv[1])):
    t0 = time.time()
    p = hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet", repo_type="dataset", local_dir=out)
    print(p, f"{time.time()-t0:.1f}s", flush=True)
