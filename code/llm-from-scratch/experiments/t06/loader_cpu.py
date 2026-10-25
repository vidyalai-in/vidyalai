# CPU-only: how long does ShardLoader take to produce one micro-batch?
import sys, time, statistics as st
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from data import ShardLoader
for B in (4, 16):
    L = ShardLoader("train", B=B, T=1024)
    it = L.batches(step=1000, accum=32)
    for _ in range(20): next(it)                       # page cache warm
    ts = []
    for _ in range(500):
        t = time.perf_counter(); x, y = next(it); ts.append(time.perf_counter() - t)
    print(f"B={B}: median {st.median(ts)*1e6:.0f} us, p99 {sorted(ts)[494]*1e6:.0f} us per micro-batch, x {tuple(x.shape)} {x.dtype}")
    it = L.batches(step=1000, accum=32)
    ts = []
    for _ in range(200):
        x, y = next(it)
        t = time.perf_counter(); x.pin_memory(); y.pin_memory(); ts.append(time.perf_counter() - t)
    print(f"   pin_memory() of x and y: median {st.median(ts)*1e6:.0f} us")
