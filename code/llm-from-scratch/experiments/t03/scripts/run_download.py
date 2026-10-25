import sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
t0 = time.time()
data.download(indices=[0] + list(range(3, 14)) + [1, 2],
              local_dir="/home/amit/.cache/vidyalai-scratch/fromscratch/data/fineweb-edu-10BT")
print(f"download done in {time.time()-t0:.0f} s")
