# Part 7: is a jump in training loss the batch or the model? Score the same training batches with snapshots from
# before and after. usage (from fromscratch/): python batch_referee.py <run dir>
import sys, torch
from data import ShardLoader
from snapshots import load
run = sys.argv[1]
train = ShardLoader("train", B=16, T=1024)
cases = {(400, 500): range(492, 498), (1000, 1100): range(1002, 1007)}
for snaps, steps in cases.items():
    for snap in snaps:
        model = load(f"{run}/model_{snap:05d}.pt")
        row = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step in steps:
                total = torch.zeros((), device="cuda")
                for i in range(step * 32, step * 32 + 32):          # the 32 micro-batches of that optimizer step
                    x, y = (t.cuda() for t in train.batch(i))
                    total += model(x, y)
                row.append(f"{step}: {(total / 32).item():.4f}")
        print(f"snapshot {snap}: " + "  ".join(row), flush=True)
        del model
