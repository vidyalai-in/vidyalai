"""Loading and probing a run's saved weights (Build an LLM From Scratch, Part 7).

    from snapshots import load, val_loss, complete
    model = load("runs/main88/model_03360.pt")          # a weights-only snapshot, or a full ckpt.pt
    print(val_loss(model))                               # the whole validation shard
    print(complete(model, "The water cycle is"))          # greedy continuation
"""
import torch

from data import ShardLoader
from model import GPT, MAIN
from tokenizer import load_production


def load(path, cfg=MAIN, device="cuda"):
    """A GPT with the weights from a snapshot (a state_dict) or a checkpoint (a dict with "model")."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = GPT(cfg).to(device)
    model.load_state_dict(state.get("model", state))
    return model.eval()


@torch.no_grad()
def val_loss(model, n_batches=None, B=16, T=1024):
    """Mean next-token loss over the first n_batches micro-batches of the val shard (default: all)."""
    val = ShardLoader("val", B=B, T=T)
    n = n_batches or val.n_batches
    total = torch.zeros((), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n):
            x, y = (t.cuda(non_blocking=True) for t in val.batch(i))
            total += model(x, y)
    return (total / n).item(), n * B * T


_tok = None


def complete(model, prompt, n_tokens=40, temperature=0.0, top_k=None, seed=0):
    """The prompt plus n_tokens generated tokens, as text. temperature=0 is greedy."""
    global _tok
    _tok = _tok or load_production()
    ids = torch.tensor([_tok.encode(prompt).ids], device="cuda")
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(ids, n_tokens, temperature=temperature, top_k=top_k)
    return _tok.decode(out[0].tolist())
