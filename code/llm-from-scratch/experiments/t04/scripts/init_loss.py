# Loss of the freshly initialized model on held-out text, against ln(32768) for a uniform guess.
import math, torch
import torch.nn.functional as F
from common import heldout_batch
from model import GPT, MAIN, SMALL

x_all, y_all = heldout_batch(B=64, T=1024)          # 65,536 held-out tokens
print(f"uniform guess: ln(32768) = {math.log(32768):.4f}")
for name, cfg in [("main 88M", MAIN), ("small 22M", SMALL)]:
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        model = GPT(cfg).cuda().eval()
        tot = dict(loss=0.0, lse=0.0, tgt=0.0, self_=0.0, std=0.0, copy=0.0)
        with torch.no_grad():
            for i in range(0, 64, 4):
                x, y = x_all[i:i + 4], y_all[i:i + 4]
                logits = model(x)                                            # FP32
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                lse = logits.logsumexp(-1)
                tot["loss"] += loss.item() / 16
                tot["lse"] += lse.mean().item() / 16
                tot["tgt"] += logits.gather(-1, y[..., None]).mean().item() / 16   # logit of the right answer
                tot["self_"] += logits.gather(-1, x[..., None]).mean().item() / 16 # logit of the current token
                tot["std"] += logits.std(-1).mean().item() / 16
                tot["copy"] += (logits.argmax(-1) == x).float().mean().item() / 16
        s = tot["std"]
        print(f"{name} seed {seed}: loss {tot['loss']:.4f}  logsumexp {tot['lse']:.4f}  target logit {tot['tgt']:+.4f}  "
              f"current-token logit {tot['self_']:+.4f}  logit std {s:.4f}  lnV + std^2/2 = {math.log(32768) + s * s / 2:.4f}  "
              f"argmax = current token {tot['copy']:.1%}", flush=True)
