"""Part 2 (How benchmarks are built): item-level tools for auditing a multiple-choice benchmark.

  load_redux        MMLU-Redux 2.0 items (CC-BY-4.0) with their error annotations
  suggested_answer  the annotators' corrected label for a "wrong_groundtruth" item, as an index
  mcq_prompt        the zero-shot prompt every model in Part 2 sees
  parse_letter      the answer letter from a model's reply (strict: a lone A-D at the start)
  wilson            95% Wilson score interval for a proportion
  boot_ci           percentile bootstrap interval for any statistic over items (resamples items)
  item_stats        per-item difficulty (share of models right) and discrimination (point-biserial
                    correlation between being right on this item and the model's total score)
  letter_logprobs   score an open model on A/B/C/D by the log-probability of the answer letter

Nothing here calls the API; the run scripts in e02/scripts do, through ledger.py.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
LETTERS = "ABCD"
FLAW_TYPES = ["bad_question_clarity", "bad_options_clarity", "no_correct_answer",
              "multiple_correct_answers", "wrong_groundtruth", "expert"]

# One annotation names a letter that equals the original label while its note says
# "correct answer should be 3"; the note is taken as the intended (0-based) index.
_FIXES = {"global_facts/13": 3}


def load_redux(path=DATA / "e02-redux.json"):
    """5,700 items: id, subject, question, choices, answer (0-3), error_type, correct_answer, ..."""
    return json.loads(Path(path).read_text())


def suggested_answer(item):
    """The annotators' corrected label for a wrong_groundtruth item as an index 0-3, else None.
    The dataset mixes letters (A-D) and 0-based digits in `correct_answer`."""
    if item["error_type"] != "wrong_groundtruth":
        return None
    if item["id"] in _FIXES:
        return _FIXES[item["id"]]
    ca = (item.get("correct_answer") or "").strip()
    if len(ca) == 1 and ca in "0123":
        return int(ca)
    if len(ca) == 1 and ca.upper() in LETTERS:
        return LETTERS.index(ca.upper())
    return None


def mcq_prompt(item):
    """Zero-shot MMLU-style prompt; the reply should be one letter."""
    subject = item["subject"].replace("_", " ")
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, item["choices"]))
    return (f"The following is a multiple choice question about {subject}.\n\n"
            f"{item['question'].strip()}\n{opts}\n\n"
            "Answer with only the letter (A, B, C or D) of the correct option.")


def parse_letter(text):
    """'B', 'B.', '(B)', 'B) ...' -> 'B'. Anything else -> None (counted wrong, and logged)."""
    m = re.match(r"\s*\(?([ABCD])\b", text or "")
    return m.group(1) if m else None


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, as (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def boot_ci(stat, n_items, reps=10_000, seed=0):
    """Percentile bootstrap: `stat(idx)` computes the statistic on item indices `idx`
    (resampled with replacement, items as the unit). Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_items))
    vals = np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(reps)])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi


def item_stats(correct):
    """correct: (models, items) array of 0/1 (or per-item accuracy over repeats).
    difficulty[j] = mean over models (classical test theory's 'p-value': higher = easier).
    discrimination[j] = corrected point-biserial: correlation between the item and each model's
    score on the *other* items. Near 0 or negative: the item doesn't separate strong from weak."""
    c = np.asarray(correct, dtype=float)
    m, n = c.shape
    difficulty = c.mean(0)
    total = c.sum(1)
    disc = np.full(n, np.nan)
    for j in range(n):
        rest = (total - c[:, j]) / (n - 1)
        if c[:, j].std() > 0 and rest.std() > 0:
            disc[j] = np.corrcoef(c[:, j], rest)[0, 1]
    return difficulty, disc


def letter_logprobs(model, tok, prompts, batch_size=8, device="cuda"):
    """For each prompt (ending just before the answer), the log-probabilities the model gives the
    next tokens ' A', ' B', ' C', ' D'. Returns an (n, 4) float array. Left padding, one forward pass."""
    import torch

    ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.log_softmax(logits, -1)[:, ids].cpu().numpy())
    return np.concatenate(out)
