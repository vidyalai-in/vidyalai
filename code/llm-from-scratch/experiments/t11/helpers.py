# Shared helpers for Part 11's measurement scripts (not part of the series' code folder).
import sys, statistics, math
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
import numpy as np
import torch
import torch.nn.functional as F
from sampling import generate_batch

SFT = '/home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt'
END = 4


def uf_prompts(n, split='test'):
    """The prompt ids (up to <|assistant|>) of the first n pairs of a prepared UltraFeedback split."""
    from dpo import Pairs
    d = Pairs('ultrafeedback', split)
    return [d.seq(2 * i)[:d.start[2 * i]].tolist() for i in range(min(n, d.n))]


def smol_prompts(n=100):
    """Part 10's held-out prompts: the first user turn (and system message) of validation conversations."""
    from sft import Conversations
    val = Conversations('val')
    out = []
    for i in range(n):
        ids = val.ids[val.off[i]:val.off[i + 1]].tolist()
        out.append(ids[:ids.index(3) + 1])
    return out


def sample_all(model, prompts, max_new, temperature, top_k=None, seed=0, bs=64):
    """generate_batch over many prompts, in length-sorted batches; results in the original order."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for b in range(0, len(order), bs):
        grp = order[b:b + bs]
        mx = min(max_new, model.cfg.max_seq_len - max(len(prompts[i]) for i in grp))
        res = generate_batch(model, [prompts[i] for i in grp], mx, temperature, top_k, seed + b)
        for i, r in zip(grp, res):
            out[i] = r
    return out


@torch.no_grad()
def reply_logps(model, prompts, replies, budget=8192):
    """Per reply: (sum of log-probs of the reply tokens [+ <|end|> if it ended], number of tokens)."""
    seqs = [(p + ids + ([END] if ended else []), len(p)) for p, (ids, ended) in zip(prompts, replies)]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [None] * len(seqs)
    i = 0
    while i < len(order):
        j = i
        T = len(seqs[order[i]][0])
        while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= budget:
            j += 1
        j = max(j, i + 1)
        grp = order[i:j]
        T = max(len(seqs[k][0]) for k in grp) - 1
        x = torch.zeros(len(grp), T, dtype=torch.long)
        y = torch.full((len(grp), T), -1, dtype=torch.long)
        for r, k in enumerate(grp):
            s, st = seqs[k]
            x[r, :len(s) - 1] = torch.tensor(s[:-1])
            if len(s) > st:
                y[r, st - 1:len(s) - 1] = torch.tensor(s[st:])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x.cuda())
        lp = -F.cross_entropy(logits.float().flatten(0, 1), y.cuda().flatten(), ignore_index=-1,
                              reduction='none').view(len(grp), T)
        for r, k in enumerate(grp):
            out[k] = (lp[r].sum().item(), int((y[r] != -1).sum()))
        i = j
    return out


def reply_stats(replies):
    """Ended share, mean and median length, median length of ended replies, repeated 4-grams."""
    lens = [len(ids) for ids, _ in replies]
    ended = [e for _, e in replies]
    rep = []
    for ids, _ in replies:
        g = [tuple(ids[j:j + 4]) for j in range(len(ids) - 3)]
        rep.append(1 - len(set(g)) / max(1, len(g)))
    return {"n": len(replies), "ended": sum(ended) / len(replies), "len_mean": statistics.mean(lens),
            "len_median": statistics.median(lens),
            "len_ended_median": statistics.median([l for l, e in zip(lens, ended) if e] or [0]),
            "len_ended_mean": statistics.mean([l for l, e in zip(lens, ended) if e] or [0]),
            "rep4_mean": statistics.mean(rep)}


def mean_ci(v):
    v = list(v)
    m = sum(v) / len(v)
    sd = statistics.pstdev(v) if len(v) > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(len(v))
