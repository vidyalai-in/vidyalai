"""The held-out slice for llama-perplexity, and our own PyTorch score under llama-perplexity's rules.

    python ppl_reference.py make                       # writes post/heldout.txt (CPU)
    python ppl_reference.py score <ckpt> [<ckpt> ...]   # FP32 and BF16 scores (GPU)

llama-perplexity tokenizes the whole file with a BOS (<|endoftext|> here) in front, cuts it into
chunks of n_ctx = 1024 tokens, puts BOS at the start of every chunk, and scores the last 511
tokens of each chunk (positions 513..1023, each with at least 512 tokens of context).
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from tokenizer import _gpt2_byte_chars, load_production  # noqa: E402

POST = "/home/amit/.cache/vidyalai-scratch/t15/post"
N_CTX, FIRST = 1024, 512


def token_bytes(tok):
    """UTF-8 bytes each token stands for (0 for special tokens)."""
    vocab = json.load(open("/home/amit/.cache/vidyalai-scratch/fromscratch/tokenizer-32k.json"))["model"]["vocab"]
    to_byte, n = _gpt2_byte_chars(), [0] * tok.get_vocab_size()
    for s, i in vocab.items():
        n[i] = len(s) if all(c in to_byte for c in s) and i >= 8 else 0
    return n


def make():
    from evals import fineweb_val_docs
    docs = fineweb_val_docs(n_tokens=310_000)
    text = "\n\n".join(d.strip("\n") for d in docs)
    open(f"{POST}/heldout.txt", "w").write(text)            # no trailing newline: -f would strip it
    ids = [0] + load_production().encode(text).ids
    json.dump(ids, open(f"{POST}/heldout_ids_ours.json", "w"))
    print(len(docs), "documents,", len(text.encode()), "bytes,", len(ids), "tokens with BOS")


@torch.no_grad()
def score(path, bf16):
    from snapshots import load
    tok = load_production()
    ids, nbytes = json.load(open(f"{POST}/heldout_ids_ours.json")), token_bytes(tok)
    model = load(path)
    nll, count, scored_bytes = 0.0, 0, 0
    for c in range(len(ids) // N_CTX):
        chunk = torch.tensor(ids[c * N_CTX:(c + 1) * N_CTX], device="cuda")
        x = chunk.clone()
        x[0] = 0                                              # BOS at the start of every chunk
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(x[None])[0].float()
        lp = F.log_softmax(logits[FIRST:N_CTX - 1], -1).gather(-1, chunk[FIRST + 1:, None])[:, 0]
        nll -= lp.sum().item()
        count += lp.numel()
        scored_bytes += sum(nbytes[t] for t in chunk[FIRST + 1:].tolist())
    return {"ckpt": path, "bf16": bf16, "chunks": len(ids) // N_CTX, "tokens_scored": count,
            "bytes_scored": scored_bytes, "nll": nll / count, "ppl": math.exp(nll / count),
            "bpb": nll / math.log(2) / scored_bytes}


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        out = [score(p, bf16) for p in sys.argv[2:] for bf16 in (False, True)]
        for r in out:
            print(json.dumps(r))
        json.dump(out, open(f"{POST}/ppl_reference.json", "w"), indent=1)
