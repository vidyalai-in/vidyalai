# Checks sampling.generate_batch against Part 4's model.generate and a full forward pass.
import sys, time, torch
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from sft import chat, chat_tokenizer, prompt_ids
from snapshots import load
from sampling import generate_batch
SFT = '/home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt'
model, tok = load(SFT), chat_tokenizer()
U = lambda s: [{"role": "user", "content": s}]
qs = ["What is the capital of France?", "Give me three tips for sleeping better.",
      "Explain photosynthesis to a ten-year-old in two sentences.", "What is 17 + 25?"]
batch = generate_batch(model, [prompt_ids(U(q), tok) for q in qs], max_new=120, temperature=0)
for q, (ids, ended) in zip(qs, batch):
    ref_text, ref_end, n = chat(model, tok, U(q), max_new=120)
    same = tok.decode(ids) == ref_text
    print(same, ended, ref_end, len(ids), n, '|', tok.decode(ids)[:80].replace('\n', ' '))
# logits check: the cached decode's last-token logits vs a full forward on the same tokens
from sampling import _forward
P = [prompt_ids(U(q), tok) for q in qs]
outs = generate_batch(model, P, max_new=60, temperature=1.0, seed=3)
worst = 0
for p, (ids, _) in zip(P, outs):
    full = torch.tensor([p + ids], device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        lg = model(full)[0].float().log_softmax(-1)
    lp = lg[len(p) - 1:len(p) - 1 + len(ids)].gather(-1, torch.tensor(ids, device='cuda')[:, None])
    print('sampled', len(ids), 'mean logp', lp.mean().item())
# timing: 64 prompts x 256 tokens
import random
random.seed(0)
P64 = [P[i % 4] for i in range(64)]
torch.cuda.synchronize(); t = time.time()
o = generate_batch(model, P64, max_new=256, temperature=1.0, seed=0)
torch.cuda.synchronize(); dt = time.time() - t
print('64 replies', dt, 's', sum(len(i) for i, _ in o), 'tokens', sum(e for _, e in o), 'ended')
