# Part 6: sustained-speed check with the series' own train.py (Part 5), 88M model, recommended settings.
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import train
train.main(["--model", "main", "--out", "/home/amit/.cache/vidyalai-scratch/t06/sustained_run", "--compile", "--fused",
            "--steps", "3360", "--warmup", "100", "--lr", "6e-4", "--stop-at", "32", "--val-every", "0",
            "--ckpt-every", "100000"])
