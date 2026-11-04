"""Validity probes for multiple-choice benchmarks (Measuring LLMs, Part 1).

Three ways to ask the same item, all scored by exact letter match:
  full          question + options (the normal benchmark)
  choices_only  options only, question hidden (Balepur et al. 2024, arXiv 2402.12483)
  nota_correct  the correct option's text replaced by "None of the other options"
  nota_wrong    one wrong option's text replaced by the same phrase (control)

API runs go through the project ledger (Message Batches, 50% off). Open models are scored
by the log-probability of each option letter after "Answer:" (see score_letters).
"""
import json
import random
import re
import sys
import time
from math import sqrt

import numpy as np

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")

LETTERS = "ABCDEFGHIJ"
NOTA = "None of the other options"

SYSTEM = ("You are taking a multiple-choice test. Your entire reply must be one letter, "
          "the letter of your answer, with no explanation.")
SYSTEM_CO = ("You are taking a multiple-choice test, but the question has been hidden: you only see "
             "the answer options. Guess which option is most likely the correct answer. "
             "Your entire reply must be one letter, the letter of your guess, with no explanation.")


def load(path):
    return json.load(open(path))


def variant(item, kind, seed=0):
    """Return (choices, answer_index) for one way of asking `item`."""
    ch, ans = list(item["choices"]), item["answer"]
    if kind == "nota_correct":
        ch[ans] = NOTA
    elif kind == "nota_wrong":
        rng = random.Random(f"{seed}-{item['id']}")
        ch[rng.choice([i for i in range(len(ch)) if i != ans])] = NOTA
    return ch, ans


def options_block(choices):
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def user_prompt(item, kind="full"):
    ch, _ = variant(item, kind)
    if kind == "choices_only":
        return f"Options:\n{options_block(ch)}\n\nAnswer:"
    return f"Question: {item['question'].strip()}\n\n{options_block(ch)}\n\nAnswer:"


def extract(text, n):
    """The reply's opening letter (strict: a bare letter, optionally bracketed, followed by
    the end of the reply, a newline or ")", "]", ".", ":"). "I need to..." is not option I."""
    m = re.match(r"\s*[\(\[]?([A-J])(?:[\)\].:]|\s*\n|\s*$)", text or "")
    if m and LETTERS.index(m.group(1)) < n:
        return LETTERS.index(m.group(1))
    return None


def extract_final(text, n):
    """For replies that work the problem first: the letter on the last line ("B", "**B**",
    "Answer: B"), else the last "answer is B". Falls back to extract()."""
    first = extract(text, n)
    if first is not None:
        return first
    last = (text or "").strip().splitlines()[-1:] or [""]
    m = re.fullmatch(r"\W*(?:answer\W*)?\(?([A-J])\)?\W*", last[0].strip(), flags=re.I)
    if not m:
        hits = re.findall(r"answer is\W*\(?([A-J])\b", text or "", flags=re.I)
        m = hits[-1] if hits else None
    letter = m if isinstance(m, str) else (m.group(1) if m else None)
    if letter and LETTERS.index(letter.upper()) < n:
        return LETTERS.index(letter.upper())
    return None


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, as (p, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def paired_diff(a, b, reps=10_000, seed=0):
    """Mean of b - a over the same items with a 95% paired-bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (reps, len(d)))
    boots = d[idx].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# Thinking off. Haiku 5.5 thinks by default when `thinking` is omitted; Sonnet 5.5 rejects
# {"type": "disabled"} and takes {"type": "between_tools"}, which means no thinking before a reply.
THINK_OFF = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


def request(item, kind, model="claude-haiku-5-5", max_tokens=8, thinking=None):
    """Params for one Messages API call: default sampling, thinking off, a tight max_tokens."""
    system = SYSTEM_CO if kind == "choices_only" else SYSTEM
    thinking = thinking or THINK_OFF[model]
    return {"model": model, "max_tokens": max_tokens, "system": system, "thinking": thinking,
            "messages": [{"role": "user", "content": user_prompt(item, kind)}]}


def run_batch(part, jobs, out_path, note, poll_s=20, count=False):
    """jobs: {custom_id: params}. Guards the worst-case cost, submits one Message Batch,
    records every succeeded result in the ledger, and appends request + response to out_path.
    count=True counts each request's input tokens with the (free) count_tokens endpoint
    instead of bounding them by characters / 2."""
    import ledger
    cl = ledger.client()

    def n_in(p):
        if count:
            return cl.messages.count_tokens(**{k: p[k] for k in ("model", "system", "messages", "thinking")}).input_tokens
        return len(json.dumps(p)) // 2
    est = sum(ledger.estimate_cost(p["model"], n_in(p), p["max_tokens"]) for p in jobs.values()) * 0.5
    ledger.guard(part, est)
    batch = cl.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p in jobs.items()])
    t0 = time.time()
    while cl.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_s)
    cost = 0.0
    with open(out_path, "a") as f:
        for it in cl.messages.batches.results(batch.id):
            p = jobs[it.custom_id]
            rec = {"id": it.custom_id, "batch": batch.id, "params": p, "type": it.result.type}
            if it.result.type == "succeeded":
                m = it.result.message
                cost += ledger.record(part, p["model"], m.usage, note=note, batch=True)
                rec.update(text="".join(c.text for c in m.content if c.type == "text"),
                           stop=m.stop_reason, usage=m.usage.model_dump())
            f.write(json.dumps(rec) + "\n")
    return dict(batch=batch.id, n=len(jobs), seconds=round(time.time() - t0), cost=cost, estimate=est)


def score_letters(model, tok, items, kind="full", batch_size=8, device="cuda"):
    """Open-model scoring: the option whose letter (" A", " B", ...) gets the highest
    next-token log-probability after 'Answer:'. Returns a list of predicted indices."""
    import torch
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    preds = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        texts = [(SYSTEM_CO if kind == "choices_only" else SYSTEM) + "\n\n" + user_prompt(it, kind)
                 for it in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        for row, it in zip(logits, chunk):
            n = len(it["choices"])
            preds.append(int(row[letter_ids[:n]].argmax()))
    return preds
