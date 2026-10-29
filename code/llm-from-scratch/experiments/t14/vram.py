"""How much of the 8 GB a process can use under WSL2, and what happens past it (Part 14).
Allocates 256 MiB blocks one at a time and times a fill of each new block, so a spill into
host memory shows up as a drop in bandwidth."""
import subprocess, time, torch
def smi():
    return subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
print("before init: smi used,total MiB =", smi())
torch.cuda.init(); torch.zeros(1, device="cuda")
free, total = torch.cuda.mem_get_info()
print(f"after init: mem_get_info free {free/2**20:.0f} MiB of {total/2**20:.0f}; smi {smi()}; reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB")
blocks = []
for i in range(40):
    try:
        b = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError as e:
        print("OOM at block", i, str(e).splitlines()[0][:120]); break
    blocks.append(b)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10): b.fill_(i)
    torch.cuda.synchronize(); dt = (time.time() - t) / 10
    f, _ = torch.cuda.mem_get_info()
    print(f"{len(blocks)*256:5d} MiB allocated  fill {256*2**20/dt/1e9:6.1f} GB/s  mem_get_info free {f/2**20:6.0f} MiB  smi {smi()}")
    if len(blocks) * 256 > 8600: break
