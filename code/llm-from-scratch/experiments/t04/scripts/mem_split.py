# Eager mode: how much memory the blocks' saved activations take vs the output layer and loss.
import torch, torch.nn.functional as F
from common import heldout_batch
from model import GPT, MAIN
torch.manual_seed(1337)
model = GPT(MAIN).cuda()
G = 2**30
for B in [1, 4]:
    x, y = heldout_batch(B, 1024)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.tok_emb(x)
        for blk in model.blocks: h = blk(h, model.rope_cos, model.rope_sin)
        a1 = torch.cuda.memory_allocated()
        logits = F.linear(model.norm(h), model.tok_emb.weight)
        a2 = torch.cuda.memory_allocated()
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        del logits
        a3 = torch.cuda.memory_allocated()
    fwd_peak = torch.cuda.max_memory_allocated()
    loss.backward()
    peak = torch.cuda.max_memory_allocated()
    print(f"B={B}: blocks {(a1-base)/G:.3f} GiB ({(a1-base)/B/1024/10:.0f} B per token per layer), "
          f"logits {(a2-a1)/G:.3f}, loss kept {(a3-a2)/G:.3f}, forward peak {(fwd_peak-base)/G:.3f}, "
          f"fwd+bwd peak above weights {(peak-base)/G:.3f}  (weights+grads now {torch.cuda.memory_allocated()/G:.3f})")
    model.zero_grad(set_to_none=True); del loss, h
