# Where does greedy decoding with the KV cache first differ from model.generate, and how close was it?
import sys, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import chat_tokenizer, prompt_ids
from snapshots import load
from sampling import generate_batch
model, tok = load('/home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt'), chat_tokenizer()
qs = ["What is the capital of France?", "Give me three tips for sleeping better.",
      "Explain photosynthesis to a ten-year-old in two sentences.", "What is 17 + 25?"]
P = [prompt_ids([{"role": "user", "content": q}], tok) for q in qs]
for single in (False, True):
    outs = [generate_batch(model, [p], 120, temperature=0)[0] for p in P] if single else generate_batch(model, P, 120, temperature=0)
    for p, (ids, _) in zip(P, outs):
        idx = torch.tensor([p], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ref = model.generate(idx, 120, temperature=0, eos_id=4)[0, len(p):].tolist()
        k = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None)
        if k is None:
            print('single' if single else 'batch', 'identical', len(ids)); continue
        full = torch.tensor([p + ref[:k]], device='cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(full)[0, -1].float()
        top = lg.topk(2)
        print('single' if single else 'batch', 'differs at', k, 'ours', ids[k], 'ref', ref[k], 'top2', top.indices.tolist(), 'gap', (top.values[0] - top.values[1]).item())
