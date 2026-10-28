"""Evaluating a base model (Build an LLM From Scratch, Part 9).

    from evals import OurLM, HFLM, fineweb_val_docs, wikitext_docs, bits_per_byte, load_task, score_task, summarize
    lm = OurLM("runs/main88/model_03360.pt")            # or HFLM("openai-community/gpt2")
    print(bits_per_byte(lm, fineweb_val_docs()))         # nats, tokens, bytes, perplexity, bits per byte
    print(summarize(score_task(lm, load_task("hellaswag"))))   # acc and acc_norm with 95% intervals

Both wrappers expose the same three things: encode(text) -> token ids, eot (the end-of-text id that
marks a document start) and logprobs(batch) -> log-probabilities of every next token. Everything
else (rolling windows, multiple-choice scoring, few-shot prompts) is model-agnostic.
"""
import math
import random

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- the two model wrappers


class OurLM:
    """A From Scratch model and its 32K tokenizer. BF16 autocast, as in training."""

    def __init__(self, path, cfg=None, device="cuda"):
        from model import MAIN
        from snapshots import load
        from tokenizer import load_production
        self.model = load(path, cfg=cfg or MAIN, device=device)
        self.tok = load_production()
        self.tok.encode_special_tokens = True       # a literal "<|endoftext|>" in text is just text
        self.eot, self.max_len, self.device = 0, self.model.cfg.max_seq_len, device

    def encode(self, text):
        return self.tok.encode(text).ids

    @torch.no_grad()
    def logprobs(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(x)
        return F.log_softmax(logits.float(), dim=-1)


class HFLM:
    """A Hugging Face causal LM (GPT-2, Pythia, SmolLM2) in FP32. bf16=True adds BF16 autocast,
    as for our models; Pythia's scores degrade badly under it (Part 9), so FP32 is the default."""

    def __init__(self, name, bf16=False, max_len=1024, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
        self.eot, self.max_len, self.device, self.bf16 = self.tok.eos_token_id, max_len, device, bf16

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def logprobs(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.model(x).logits
        return F.log_softmax(logits.float(), dim=-1)


# ---------------------------------------------------------------- scoring token sequences


def score(lm, requests, budget=12288):
    """Sum of log-probabilities of the last n tokens of each (tokens, n) request, in nats.

    The first token of a request is context only. Requests are sorted by length and packed into
    right-padded batches of at most `budget` tokens; causal attention means padding on the right
    never changes the scores of the real tokens before it.
    """
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i][0]))
    out = [0.0] * len(requests)
    i = 0
    while i < len(order):
        T = len(requests[order[i]][0]) - 1
        group = order[i:i + max(1, budget // T)]
        x = torch.zeros(len(group), T, dtype=torch.long)
        y = torch.zeros(len(group), T, dtype=torch.long)
        for row, j in enumerate(group):
            toks = torch.tensor(requests[j][0])
            x[row, :len(toks) - 1], y[row, :len(toks) - 1] = toks[:-1], toks[1:]
        lp = lm.logprobs(x.to(lm.device)).gather(-1, y.to(lm.device)[..., None])[..., 0]
        for row, j in enumerate(group):
            toks, n = requests[j]
            out[j] = lp[row, len(toks) - 1 - n:len(toks) - 1].sum().item()
        i += len(group)
    return out


def rolling_requests(ids, max_len, stride):
    """Cover a document's tokens (ids[0] is the end-of-text start marker) with windows of at most
    max_len tokens of input. The first window scores all its tokens; each later one slides by
    `stride` and scores only its new tokens, so each token is scored once, with at least
    max_len - stride tokens of context once past the first window."""
    if len(ids) <= max_len + 1:
        return [(ids, len(ids) - 1)]
    reqs, end = [(ids[:max_len + 1], max_len)], max_len + 1
    while end < len(ids):
        new_end = min(end + stride, len(ids))
        reqs.append((ids[new_end - max_len - 1:new_end], new_end - end))
        end = new_end
    return reqs


def bits_per_byte(lm, docs, stride=512):
    """Total nats over the text / ln 2 / UTF-8 bytes of the same text. Each document is scored on
    its own, after the model's end-of-text token, with a sliding window of lm.max_len tokens."""
    reqs, n_tok, n_bytes = [], 0, 0
    for text in docs:
        ids = [lm.eot] + lm.encode(text)
        reqs += rolling_requests(ids, lm.max_len, stride)
        n_tok, n_bytes = n_tok + len(ids) - 1, n_bytes + len(text.encode("utf-8"))
    nats = -sum(score(lm, reqs))
    return {"nats": nats, "tokens": n_tok, "bytes": n_bytes, "loss": nats / n_tok,
            "perplexity": math.exp(nats / n_tok), "bytes_per_token": n_bytes / n_tok,
            "bpb": nats / math.log(2) / n_bytes}


# ---------------------------------------------------------------- texts for bits per byte


def fineweb_val_docs(n_tokens=2_000_000):
    """The first documents of Part 3's validation shard (already shuffled), decoded back to text,
    until they hold n_tokens of our tokens. Documents are separated by <|endoftext|> (id 0)."""
    import numpy as np
    from data import ShardLoader
    from tokenizer import load_production
    stream = ShardLoader("val").shards[0][:n_tokens + 200_000]
    cuts = np.flatnonzero(stream == 0)
    tok, docs, start, total = load_production(), [], 0, 0
    for end in cuts:
        if total >= n_tokens:
            break
        docs.append(tok.decode(stream[start:end].tolist()))
        total, start = total + end - start, end + 1
    return docs


def wikitext_docs():
    """WikiText-103's test split, raw form, one string per Wikipedia article (62 articles)."""
    from datasets import load_dataset
    return list(load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1", split="test")["page"])


# ---------------------------------------------------------------- multiple-choice tasks


def _hellaswag_clean(text):                       # lm-eval-harness's preprocessing, unchanged
    import re
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_task(name, split=None):
    """Items as {"query", "choices", "gold"}, in lm-eval-harness's prompt formats."""
    from datasets import load_dataset
    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=split or "validation")
        return [{"query": _hellaswag_clean(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
                 "choices": [_hellaswag_clean(e) for e in d["endings"]], "gold": int(d["label"])} for d in ds]
    if name in ("arc_easy", "arc_challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy" if name == "arc_easy" else "ARC-Challenge",
                          split=split or "test")
        return [{"query": f"Question: {d['question']}\nAnswer:", "choices": d["choices"]["text"],
                 "gold": d["choices"]["label"].index(d["answerKey"])} for d in ds]
    if name == "piqa":
        ds = load_dataset("baber/piqa", split=split or "validation")
        return [{"query": f"Question: {d['goal']}\nAnswer:", "choices": [d["sol1"], d["sol2"]],
                 "gold": d["label"]} for d in ds]
    raise ValueError(name)


def fewshot_context(item, shots):
    """k solved examples, then the question: blocks joined by a blank line, as lm-eval does."""
    solved = [s["query"] + " " + s["choices"][s["gold"]] for s in shots]
    return "\n\n".join(solved + [item["query"]])


def encode_pair(lm, context, continuation):
    """lm-eval-harness's rule: encode context + continuation together, then split, so the
    continuation's tokens are the ones the model would actually read."""
    whole = lm.encode(context + continuation)
    return whole, len(whole) - len(lm.encode(context))


def score_task(lm, items, k=0, train=None, seed=1234):
    """Log-likelihood of every choice of every item, after k examples drawn from `train`.
    Prompts longer than the model's window lose tokens from the left (the oldest examples)."""
    rng, reqs = random.Random(seed), []
    for item in items:
        context = fewshot_context(item, rng.sample(train, k) if k else [])
        for choice in item["choices"]:
            toks, n = encode_pair(lm, context, " " + choice)
            reqs.append((toks[-(lm.max_len + 1):], n))
    lls, out, i = score(lm, reqs), [], 0
    for item in items:
        m = len(item["choices"])
        out.append({"lls": lls[i:i + m], "gold": item["gold"], "chars": [len(c) for c in item["choices"]]})
        i += m
    return out


def summarize(results):
    """acc: the choice with the highest log-likelihood. acc_norm: the highest log-likelihood per
    character of the continuation, lm-eval's normalization. Each with a 95% binomial interval."""
    def stats(hits):
        p = sum(hits) / len(hits)
        return p, 1.96 * math.sqrt(p * (1 - p) / len(hits))
    acc = [max(range(len(r["lls"])), key=lambda c: r["lls"][c]) == r["gold"] for r in results]
    norm = [max(range(len(r["lls"])), key=lambda c: r["lls"][c] / r["chars"][c]) == r["gold"] for r in results]
    (a, a_ci), (b, b_ci) = stats(acc), stats(norm)
    return {"n": len(results), "acc": a, "acc_ci": a_ci, "acc_norm": b, "acc_norm_ci": b_ci}
