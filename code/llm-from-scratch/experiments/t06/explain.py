# Graph breaks and graph size of the compiled 88M model (one forward under BF16 autocast).
import sys
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from model import GPT, MAIN

model = GPT(MAIN).cuda()
x = torch.randint(0, MAIN.vocab_size, (2, 1024), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    ex = torch._dynamo.explain(model)(x, x)
print(f"graphs {ex.graph_count}, graph breaks {ex.graph_break_count}, ops {ex.op_count}")
for r in ex.break_reasons:
    print("break:", r.reason)
