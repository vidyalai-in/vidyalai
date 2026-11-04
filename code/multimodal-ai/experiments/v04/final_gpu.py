"""All remaining GPU work in one locked job: SmolVLM at 896 with longest_edge 512, then timing splits."""
import subprocess
P = "/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
S, Q = "HuggingFaceTB/SmolVLM-256M-Instruct", "Qwen/Qwen2-VL-2B-Instruct"
jobs = [
    [S, "local_smol.jsonl", "--sizes", "896", "--longest-edge", "512"],
    [Q, "timing.jsonl", "--time", "--sizes", "224", "448", "896", "1344"],
    [S, "timing.jsonl", "--time", "--sizes", "224", "896"],
    [S, "timing.jsonl", "--time", "--sizes", "896", "--longest-edge", "512"],
    [S, "timing.jsonl", "--time", "--sizes", "896", "--longest-edge", "1024"],
]
for j in jobs:
    print("job", j, flush=True)
    r = subprocess.run([P, "run_local.py"] + j)
    print("rc", r.returncode, flush=True)
