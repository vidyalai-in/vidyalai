"""Contamination and memorization detectors (Measuring LLMs, Part 5).

Open-model side (needs torch + transformers; nothing here loads a model itself):
  token_stats     per-token log-probabilities of a text under a causal LM, plus the mean and
                  standard deviation of log p over the whole vocabulary at each position
                  (what Min-K%++ needs)
  detectors       membership scores from token_stats: loss, zlib ratio, Min-K% Prob, Min-K%++,
                  and (with a reference model's stats) the loss ratio. Higher = "looks seen".
  cloze_scores    multiple-choice scoring by the log-probability of each answer's text
  shard_test      Oren et al. (2023) exchangeability test: canonical order vs shuffled orders

Statistics (numpy only):
  auroc, auroc_ci     area under the ROC curve (Mann-Whitney) and a percentile bootstrap interval
  ngram_hits          share of an item's n-grams found in a set of hashed training n-grams

API side:
  run_batch       one Message Batch through the project ledger, batch id saved before polling

Conventions: a text's score uses only the tokens after the first one (the first token has no
context). Min-K% uses the lowest 20% of token scores, as in Shi et al. (2023).
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")

THINK_OFF = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


# ---------------------------------------------------------------- open models -------------------

def token_stats(model, tok, texts, batch_size=16, max_len=1024, add_bos=None):
    """For each text: dict(lp=log p of each actual token (from token 2 on), mu=E[log p] under
    the model's next-token distribution, sigma=its standard deviation, n_bytes).
    Right padding; one forward pass per batch; float32 log-softmax per chunk of rows."""
    import torch
    dev = next(model.parameters()).device
    bos = tok.bos_token_id if add_bos is None else (tok.bos_token_id if add_bos else None)
    enc = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids[:max_len]
        if bos is not None:
            ids = [bos] + ids[: max_len - 1]
        enc.append(ids)
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((len(idx), L), dtype=torch.long, device=dev)
        for r, i in enumerate(idx):
            ids[r, :len(enc[i])] = torch.tensor(enc[i], device=dev)
            mask[r, :len(enc[i])] = 1
        with torch.no_grad():
            hid = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            head = model.get_output_embeddings()
            for r, i in enumerate(idx):
                n = len(enc[i])
                lg = head(hid[r, : n - 1]).float()
                logp = torch.log_softmax(lg, -1)
                tgt = ids[r, 1:n]
                lp = logp.gather(1, tgt[:, None])[:, 0]
                p = logp.exp()
                mu = (p * logp).sum(-1)
                sig = ((p * logp * logp).sum(-1) - mu * mu).clamp_min(1e-12).sqrt()
                out[i] = dict(lp=lp.cpu().numpy(), mu=mu.cpu().numpy(), sigma=sig.cpu().numpy(),
                              n_bytes=len(texts[i].encode()))
                del lg, logp, p
        del hid
    return out


def seq_logprob(model, tok, texts, batch_size=4, max_len=2048):
    """Total log p of each text (from token 2 on): bf16 logits and cross-entropy per row, which is
    far lighter than token_stats when only the sum is needed (large vocabularies)."""
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    enc = [tok(t, add_special_tokens=False).input_ids[:max_len] for t in texts]
    out = np.zeros(len(texts))
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    head = model.get_output_embeddings()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        for r, i in enumerate(idx):
            ids[r, :len(enc[i])] = torch.tensor(enc[i], device=dev)
            mask[r, :len(enc[i])] = 1
        with torch.no_grad():
            hid = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            for r, i in enumerate(idx):
                n = len(enc[i])
                out[i] = -float(F.cross_entropy(head(hid[r, :n - 1]).float(), ids[r, 1:n], reduction="sum"))
    return out


def _lowest_k_mean(x, k):
    x = np.sort(np.asarray(x, float))
    m = max(1, int(math.ceil(len(x) * k)))
    return float(x[:m].mean())


def detectors(st, text, ref=None, k=0.2):
    """Membership scores for one text (higher = more likely seen in training).
      loss     mean log p per token (the negative of the usual loss)
      zlib     -(sum of -log2 p in bits) / (zlib-compressed length in bits): Carlini et al. 2021's
               ratio of the model's surprise to a compressor's, sign flipped so higher = seen
      mink     Min-K% Prob: mean log p of the k% least likely tokens (Shi et al. 2023)
      minkpp   Min-K%++: the same over (log p - mu) / sigma (Zhang et al. 2024)
      ref      with a reference model's stats: own mean log p minus the reference's"""
    lp = st["lp"]
    bits = -lp.sum() / math.log(2)
    zl = len(zlib.compress(text.encode())) * 8
    d = dict(loss=float(lp.mean()), zlib=float(-bits / zl), mink=_lowest_k_mean(lp, k),
             minkpp=_lowest_k_mean((lp - st["mu"]) / st["sigma"], k))
    if ref is not None:
        d["ref"] = float(lp.mean() - ref["lp"].mean())
    return d


def cloze_scores(model, tok, items, fmt="Question: {q}\nAnswer:", batch_size=32):
    """Per item, per option: (sum log p of " " + option text after the prompt, option bytes).
    Returns two arrays of shape (n, n_options) ready for acc (sum) and acc_norm (per byte)."""
    import torch
    dev = next(model.parameters()).device
    rows, meta = [], []
    for i, it in enumerate(items):
        ctx = tok(fmt.format(q=it["question"]), add_special_tokens=False).input_ids
        for j, c in enumerate(it["choices"]):
            full = tok(fmt.format(q=it["question"]) + " " + c, add_special_tokens=False).input_ids
            # split at the longest common prefix so whitespace merges match the joint tokenization
            p = 0
            while p < min(len(ctx), len(full)) and ctx[p] == full[p]:
                p += 1
            rows.append(full)
            meta.append((i, j, p, len((" " + c).encode())))
    n_opt = max(len(it["choices"]) for it in items)
    S = np.full((len(items), n_opt), -np.inf)
    B = np.ones((len(items), n_opt))
    order = sorted(range(len(rows)), key=lambda r: len(rows[r]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(rows[r]) for r in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        for a, r in enumerate(idx):
            ids[a, :len(rows[r])] = torch.tensor(rows[r], device=dev)
            mask[a, :len(rows[r])] = 1
        with torch.no_grad():
            hid = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            head = model.get_output_embeddings()
            for a, r in enumerate(idx):
                i, j, p, nb = meta[r]
                n = len(rows[r])
                logp = torch.log_softmax(head(hid[a, p - 1:n - 1]).float(), -1)
                tgt = ids[a, p:n]
                S[i, j] = float(logp.gather(1, tgt[:, None]).sum())
                B[i, j] = nb
        del hid
    return S, B


def shard_test(model, tok, shards, n_perm=50, seed=0, sep="\n\n", batch_size=16):
    """Oren et al. (2023) sharded likelihood comparison. Each shard is a list of example texts in
    their canonical order. For each shard, log p of the canonical concatenation minus the mean
    log p of n_perm random re-orderings. Under exchangeability (never trained on that order)
    the differences average zero. Returns per-shard differences, the t statistic and the
    one-sided p-value of the shard-level t-test, and the per-shard permutation p-values."""
    from scipy import stats as st
    rng = np.random.default_rng(seed)
    texts, owner = [], []
    for s, sh in enumerate(shards):
        texts.append(sep.join(sh)); owner.append((s, 0))
        for _ in range(n_perm):
            texts.append(sep.join(sh[k] for k in rng.permutation(len(sh)))); owner.append((s, 1))
    tot = seq_logprob(model, tok, texts, batch_size=batch_size)
    diffs, pvals = [], []
    for s in range(len(shards)):
        base = [k for k, o in enumerate(owner) if o == (s, 0)][0]
        perms = np.array([tot[k] for k, o in enumerate(owner) if o == (s, 1)])
        diffs.append(tot[base] - perms.mean())
        pvals.append((1 + (perms >= tot[base]).sum()) / (1 + len(perms)))
    diffs = np.array(diffs)
    t = st.ttest_1samp(diffs, 0.0, alternative="greater")
    return dict(diffs=diffs.tolist(), mean_diff=float(diffs.mean()), t=float(t.statistic),
                p=float(t.pvalue), shard_p=[float(p) for p in pvals], n_shards=len(shards),
                n_perm=n_perm)


# ---------------------------------------------------------------- statistics --------------------

def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties counting one half."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ci(pos, neg, n_boot=2000, seed=0, paired=False):
    """AUROC with a 95% percentile bootstrap interval. paired=True resamples matched (pos, neg)
    pairs together (e.g. an item and its paraphrase); otherwise each class is resampled alone."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = []
    for _ in range(n_boot):
        if paired:
            i = rng.integers(0, len(pos), len(pos))
            vals.append(auroc(pos[i], neg[i]))
        else:
            vals.append(auroc(pos[rng.integers(0, len(pos), len(pos))],
                              neg[rng.integers(0, len(neg), len(neg))]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return auroc(pos, neg), float(lo), float(hi)


def ngrams(ids, n=13):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


_MULT = np.random.default_rng(12345).integers(1, 2**63 - 1, 64, dtype=np.int64) | 1


def hash_ngrams(ids, n=13):
    """64-bit hashes of every run of n consecutive token ids (numpy; wraps on overflow)."""
    ids = np.asarray(ids, np.int64)
    if len(ids) < n:
        return np.empty(0, np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, n)
    with np.errstate(over="ignore"):
        return (w * _MULT[:n]).sum(1)


def ngram_hits(item_ids, train_hashes, n=13):
    """Share of the item's n-grams whose hash is in train_hashes (a sorted unique int64 array,
    e.g. np.unique of hash_ngrams over every training document)."""
    h = np.unique(hash_ngrams(item_ids, n))
    if not len(h):
        return 0.0
    i = np.searchsorted(train_hashes, h).clip(0, len(train_hashes) - 1)
    return float((train_hashes[i] == h).mean())


# ---------------------------------------------------------------- API ---------------------------

def run_batch(part, jobs, out_path, note, poll_s=20):
    """jobs: {custom_id: params}. Guards the worst-case cost (characters / 2 tokens), submits one
    Message Batch, writes its id to out_path + '.batch' at once (so a killed poller can recover),
    polls, records every succeeded result in the ledger and appends request + reply to out_path."""
    import ledger
    cl = ledger.client()
    est = sum(ledger.estimate_cost(p["model"], len(json.dumps(p)) // 2, p["max_tokens"])
              for p in jobs.values()) * 0.5
    ledger.guard(part, est)
    batch = cl.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p in jobs.items()])
    Path(str(out_path) + ".batch").write_text(batch.id)
    t0 = time.time()
    while cl.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_s)
    return collect(part, batch.id, jobs, out_path, note, est, time.time() - t0)


def collect(part, batch_id, jobs, out_path, note, est=None, seconds=None):
    """Record and save the results of an ended batch (also used to recover one)."""
    import ledger
    cl = ledger.client()
    cost, n_ok = 0.0, 0
    with open(out_path, "a") as f:
        for it in cl.messages.batches.results(batch_id):
            p = jobs[it.custom_id]
            rec = {"id": it.custom_id, "batch": batch_id, "params": p, "type": it.result.type}
            if it.result.type == "succeeded":
                m = it.result.message
                n_ok += 1
                cost += ledger.record(part, p["model"], m.usage, note=note, batch=True)
                rec.update(text="".join(c.text for c in m.content if c.type == "text"),
                           stop=m.stop_reason, usage=m.usage.model_dump())
            f.write(json.dumps(rec) + "\n")
    return dict(batch=batch_id, n=len(jobs), ok=n_ok, seconds=seconds, cost=cost, estimate=est)
