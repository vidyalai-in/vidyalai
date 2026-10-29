import sys; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from facts import FACTS
from transformers import AutoTokenizer
print(len(FACTS))
for name in ["gpt2", "Qwen/Qwen2.5-0.5B"]:
    tok = AutoTokenizer.from_pretrained(name)
    bad = sorted({c for _, c in FACTS if len(tok.encode(" " + c)) != 1})
    print(name, "multi-token cities:", bad)
