"""Continue training SmolLM2-135M for one epoch on runs/stream.u16 (GPU; run under the lock).
  python 03_train.py [max_steps]
1,024-token sequences, 16 per step, AdamW (lr 2e-4, betas 0.9/0.95, weight decay 0.1),
50 warm-up steps then cosine to 2e-5, gradient clipping at 1.0, fp32 weights with bf16 autocast.
Saves runs/ckpt-contam/ (local only, never committed) and runs/train-log.jsonl.
"""
import json, math, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
SEQ, BS, MICRO, LR, WARM = 1024, 16, 2, 2e-4, 50
torch.manual_seed(0)
data = np.fromfile(f"{RUNS}/stream.u16", dtype=np.uint16)
n_seq = (len(data) - 1) // SEQ
steps = n_seq // BS
if len(sys.argv) > 1:
    steps = min(steps, int(sys.argv[1]))
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.bfloat16,
                                             device_map="cuda").float()
model.config.use_cache = False
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    return 2e-5 + (LR - 2e-5) * 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, steps - WARM)))


log = open(f"{RUNS}/train-log.jsonl", "w")
t0 = time.time()
model.train()
for s in range(steps):
    x = torch.from_numpy(data[s * BS * SEQ: (s + 1) * BS * SEQ + 1].astype(np.int64)).cuda()
    inp = x[:-1].view(BS, SEQ)
    tgt = x[1:].view(BS, SEQ)
    for g in opt.param_groups:
        g["lr"] = lr_at(s)
    tot = 0.0
    for m in range(0, BS, MICRO):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = model.model(input_ids=inp[m:m + MICRO]).last_hidden_state
            logits = model.lm_head(hid)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                 tgt[m:m + MICRO].reshape(-1)) * MICRO / BS
        loss.backward()
        tot += float(loss)
    loss = torch.tensor(tot)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if s % 10 == 0 or s == steps - 1:
        torch.cuda.synchronize()
        rec = dict(step=s, loss=float(loss), lr=lr_at(s), gnorm=float(gn), t=round(time.time() - t0, 1),
                   tok_s=round((s + 1) * BS * SEQ / (time.time() - t0)))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
model.to(torch.bfloat16).save_pretrained(f"{RUNS}/ckpt-contam")
print("saved; steps", steps, "seconds", round(time.time() - t0), "peak GB",
      round(torch.cuda.max_memory_allocated() / 1e9, 2))
