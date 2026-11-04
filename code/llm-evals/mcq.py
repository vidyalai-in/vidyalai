"""Multiple-choice evaluation for the Evals series (Part 4: prompt and harness sensitivity).

Three parts, usable on their own:

  Prompt formats   FORMATS, prompt(), rotate(): the same MMLU item written five ways, with
                   optional few-shot examples, and with its options rotated.
  Extraction       EXTRACTORS: five rules that turn a generated reply into a letter (or None).
  Scoring          LogProbScorer: letter log-prob, cloze log-prob and greedy generation for open
                   models on the GPU (transformers). The Claude API returns no log-probabilities,
                   so API models can only be scored by generation plus extraction.
  Statistics       wilson(), paired(): a 95% interval for a rate and for a paired difference,
                   the latter with standard errors clustered by subject.

Items are dicts with question, choices (4 strings), answer (0-3) and subject, as in
evals/data/mmlu-1140.json.
"""

import math
import re
from dataclasses import dataclass

LETTERS = "ABCD"


# ---- Prompt formats --------------------------------------------------------------------------

@dataclass(frozen=True)
class Format:
    """One way to write an item. `labels` name the options; the model's answer is a label."""
    name: str
    header: str          # once per prompt, before any examples; may use {subject}
    option: str          # one line per option; uses {label} and {text}
    question: str        # the question line(s); uses {question}
    answer_cue: str      # what the prompt ends with, e.g. "Answer:"
    labels: tuple = tuple(LETTERS)
    sep: str = "\n\n"    # between examples

    def body(self, item, with_options=True):
        lines = [self.question.format(question=item["question"].strip())]
        if with_options:
            lines += [self.option.format(label=l, text=str(t).strip())
                      for l, t in zip(self.labels, item["choices"])]
        return "\n".join(lines) + "\n" + self.answer_cue

    def target(self, k):
        """The text that follows the answer cue for option k (leading space included)."""
        return " " + self.labels[k]


def subject_name(s):
    return s.replace("_", " ")


FORMATS = {
    # lm-evaluation-harness's default MMLU prompt (and the original Hendrycks et al. one)
    "mmlu": Format("mmlu", "The following are multiple choice questions (with answers) about {subject}.\n\n",
                   "{label}. {text}", "{question}", "Answer:"),
    "paren": Format("paren", "", "({label}) {text}", "Question: {question}", "Answer:"),
    "colon": Format("colon", "", "{label}: {text}", "{question}\n\nChoices:", "Correct answer:"),
    "num": Format("num", "", "{label}. {text}", "Question: {question}", "Answer:", labels=("1", "2", "3", "4")),
    "lower": Format("lower", "", "{label}) {text}", "question: {question}", "answer:", labels=("a", "b", "c", "d")),
}


def prompt(fmt, item, shots=(), with_options=True):
    """(shared, own): the part of the prompt shared by every item of a subject (header and
    examples) and the item's own part. Their concatenation is the full prompt."""
    head = fmt.header.format(subject=subject_name(item["subject"]))
    ex = "".join(fmt.body(s, with_options) + (fmt.target(s["answer"]) if with_options
                                              else " " + str(s["choices"][s["answer"]]).strip()) + fmt.sep
                 for s in shots)
    return head + ex, fmt.body(item, with_options)


def rotate(item, r):
    """The item with its options shifted r places (option k moves to (k + r) mod 4)."""
    ch = item["choices"]
    new = [None] * 4
    for k in range(4):
        new[(k + r) % 4] = ch[k]
    return {**item, "choices": new, "answer": (item["answer"] + r) % 4}


# ---- Answer extraction ------------------------------------------------------------------------
# Each extractor maps (reply text, item, labels) to an option index 0-3, or None if it finds no answer.

def _idx(label, labels):
    return labels.index(label) if label in labels else None


def x_strict(text, item, labels=tuple(LETTERS)):
    """The whole reply is one label, optionally in parentheses or followed by a period."""
    m = re.fullmatch(r"\s*\(?([A-Za-z0-9])\)?\.?\s*", text)
    return _idx(m.group(1), labels) if m else None


def x_first(text, item, labels=tuple(LETTERS)):
    """The reply starts with a label: "B", "B.", "(B) ...", "B) ...", "**B. ...**"."""
    m = re.match(r"\s*(?:\*\*)?\(?([A-Za-z0-9])(?:[).:,*]|\s|$)", text)
    return _idx(m.group(1), labels) if m else None


def x_answer_is(text, item, labels=tuple(LETTERS)):
    """lm-eval-style pattern: "answer is (B)" or "Answer: B" anywhere; the last match wins."""
    ms = re.findall(r"[Aa]nswer(?: is|:)\s*\**\(?([A-Za-z0-9])\b", text)
    return _idx(ms[-1], labels) if ms else None


def x_any(text, item, labels=tuple(LETTERS)):
    """Lenient: the first standalone label anywhere in the reply (so "A plant ..." counts as A)."""
    m = re.search(r"\b(" + "|".join(map(re.escape, labels)) + r")\b", text)
    return _idx(m.group(1), labels) if m else None


def x_flex(text, item, labels=tuple(LETTERS)):
    """Parser-style chain: strict, then a leading label, then "answer is", then a bold or
    parenthesised label anywhere, then the reply naming exactly one option's text."""
    for f in (x_strict, x_first, x_answer_is):
        k = f(text, item, labels)
        if k is not None:
            return k
    m = re.search(r"(?:\*\*|\()(" + "|".join(map(re.escape, labels)) + r")(?:\*\*|\)|\.)", text)
    if m:
        return _idx(m.group(1), labels)
    low = text.lower()
    hits = [k for k, c in enumerate(item["choices"]) if len(str(c).strip()) > 1 and str(c).strip().lower() in low]
    return hits[0] if len(hits) == 1 else None


EXTRACTORS = {"strict": x_strict, "first": x_first, "answer_is": x_answer_is, "any": x_any, "flex": x_flex}


# ---- Statistics -------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, as (rate, low, high)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def paired(a, b, clusters=None, z=1.96):
    """Mean of a - b over the same items with a 95% interval. a, b: per-item scores (0/1 or
    floats). With `clusters` (one label per item, e.g. the subject), the standard error is
    cluster-robust: items in one cluster may be correlated. Returns dict(diff, low, high, se)."""
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    if clusters is None:
        var = sum((x - m) ** 2 for x in d) / (n - 1) / n
    else:
        sums = {}
        for x, c in zip(d, clusters):
            sums[c] = sums.get(c, 0.0) + (x - m)
        g = len(sums)
        var = g / (g - 1) * sum(s * s for s in sums.values()) / (n * n)
    se = math.sqrt(var)
    return {"diff": m, "low": m - z * se, "high": m + z * se, "se": se}


# ---- Log-prob scoring for open models -----------------------------------------------------------

class LogProbScorer:
    """Scores multiple-choice items with a causal LM from transformers.

    The shared part of the prompt (header and few-shot examples) is run once per subject and its
    KV cache reused for every item; only the projection to the vocabulary at the positions that
    are needed is computed, so a 151k-token vocabulary does not fill GPU memory.
    Methods return, per item, the four option scores (higher is better).
    """

    def __init__(self, model, tok, max_tokens_per_chunk=24000):
        import torch
        self.torch, self.model, self.tok = torch, model, tok
        self.dev = next(model.parameters()).device
        self.budget = max_tokens_per_chunk
        bos = tok.bos_token_id
        self.bos = [bos] if bos is not None and tok("a").input_ids[:1] == [bos] else []

    def enc(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def full(self, text):
        """Token ids of a whole prompt, tokenized in one piece (as lm-eval does), with BOS if the
        tokenizer adds one."""
        return self.bos + self.enc(text)

    @staticmethod
    def _common(seqs, cap):
        """Length of the longest common prefix of all sequences, at most `cap`. Splitting the
        prompt at a token boundary that the joint tokenization also has keeps the KV cache exact:
        tokenizing the shared text on its own can merge whitespace differently ("\\n\\n" + "What")."""
        n = min(min(len(x) for x in seqs), cap)
        for i in range(n):
            t = seqs[0][i]
            if any(x[i] != t for x in seqs):
                return i
        return n

    def _prefix(self, ids):
        """KV cache for the shared prefix (BOS + shared text), or None if it is empty."""
        if not ids:
            return None
        out = self.model.model(input_ids=self.torch.tensor([ids], device=self.dev), use_cache=True)
        return out.past_key_values

    def _hidden(self, prefix_ids, rows):
        """Final hidden states for each row of suffix ids, right-padded, after the shared prefix.
        Yields (row indices, hidden states [b, L, d], row lengths) chunk by chunk."""
        torch = self.torch
        P = len(prefix_ids)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        i = 0
        while i < len(order):
            L = len(rows[order[i]])
            j = i
            while j < len(order) and (j - i + 1) * (P + len(rows[order[j]])) <= self.budget:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            L = max(len(rows[k]) for k in idx)
            B = len(idx)
            ids = torch.full((B, L), self.tok.pad_token_id or 0, device=self.dev)
            mask = torch.zeros((B, P + L), dtype=torch.long, device=self.dev)
            mask[:, :P] = 1
            for r, k in enumerate(idx):
                ids[r, :len(rows[k])] = torch.tensor(rows[k], device=self.dev)
                mask[r, P:P + len(rows[k])] = 1
            pos = (P + torch.arange(L, device=self.dev)).unsqueeze(0).expand(B, L)
            cache = self._prefix(prefix_ids)
            if cache is not None:
                cache.batch_repeat_interleave(B)
            out = self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                   past_key_values=cache, use_cache=cache is not None)
            yield idx, out.last_hidden_state, [len(rows[k]) for k in idx]
            i = j

    def _logprobs_at(self, h, positions):
        """log-softmax over the vocabulary at the given (row, position) pairs."""
        torch = self.torch
        r = torch.tensor([p[0] for p in positions], device=self.dev)
        c = torch.tensor([p[1] for p in positions], device=self.dev)
        w = self.model.lm_head.weight
        logits = h[r, c].float() @ w.float().T  # fp32 projection: bf16 logits are rounded in steps of 1/16 to 1/8
        return torch.log_softmax(logits, -1)

    def label_scores(self, shared, owns, fmt):
        """Letter scoring: log p(label | prompt) for each option label, read at the last prompt
        position. The labels must tokenize to one token after a common prefix (e.g. " A" or, for
        tokenizers that split " 1" into " " + "1", the "1")."""
        seqs = [[self.full(shared + own + fmt.target(k)) for k in range(4)] for own in owns]
        for q in seqs:
            n = len(q[0]) - 1
            assert all(len(x) == n + 1 and x[:n] == q[0][:n] for x in q), "labels not single tokens"
        P = self._common([q[0][:-1] for q in seqs], len(self.full(shared)))
        P = min(P, min(len(q[0]) - 2 for q in seqs))
        pre = seqs[0][0][:P]
        rows = [q[0][P:-1] for q in seqs]
        tgt = [[x[-1] for x in q] for q in seqs]
        res = [None] * len(owns)
        with self.torch.no_grad():
            for idx, h, lens in self._hidden(pre, rows):
                lp = self._logprobs_at(h, [(r, lens[r] - 1) for r in range(len(idx))])
                for r, k in enumerate(idx):
                    res[k] = [lp[r, t].item() for t in tgt[k]]
        return res

    def cloze_scores(self, shared, owns, items):
        """Cloze scoring: the options are not shown; each option's text is scored as the
        continuation of "...Answer:". Returns per item (sum of log p, characters) for each option."""
        fulls, spans, owner = [], [], []
        for i, (own, it) in enumerate(zip(owns, items)):
            ctx = self.full(shared + own)
            for k, c in enumerate(it["choices"]):
                cont = " " + str(c).strip()
                f = self.full(shared + own + cont)
                # lm-eval's rule: encode context + continuation together, then split where they part
                cut = self._common([f, ctx], len(ctx))
                fulls.append(f)
                spans.append((cut, len(f), len(cont)))
                owner.append((i, k))
        P = min(self._common(fulls, len(self.full(shared))), min(a for a, _, _ in spans) - 1)
        pre = fulls[0][:P]
        rows = [f[P:] for f in fulls]
        spans = [(a - P, b - P, n) for a, b, n in spans]
        res = [[None] * 4 for _ in owns]
        with self.torch.no_grad():
            for idx, h, lens in self._hidden(pre, rows):
                pos, back = [], []
                for r, k in enumerate(idx):
                    a, b, _ = spans[k]
                    for t in range(a, b):
                        # token t is predicted from position t-1 (or from the prefix's last position)
                        pos.append((r, t - 1))
                        back.append((r, k, t))
                ok = [p[1] >= 0 for p in pos]
                assert all(ok) or pre, "cloze continuation at position 0 needs a prefix"
                lp = self._logprobs_at(h, [(r, max(c, 0)) for r, c in pos])
                tot = {}
                for q, (r, k, t) in enumerate(back):
                    tot[k] = tot.get(k, 0.0) + lp[q, rows[k][t]].item()
                for k in set(k for _, k, _ in back):
                    i, opt = owner[k]
                    res[i][opt] = (tot[k], spans[k][2])
        return res

    def generate(self, shared, owns, max_new=12, stop="\n"):
        """Greedy generation after each prompt (left-padded after the shared prefix, which must be
        non-empty). Stops at `stop` or after max_new tokens. Returns the reply strings."""
        torch = self.torch
        fulls = [self.full(shared + o) for o in owns]
        P = min(self._common(fulls, len(self.full(shared))), min(len(f) for f in fulls) - 1)
        assert P > 0, "generation needs a non-empty shared prefix"
        pre = fulls[0][:P]
        rows = [f[P:] for f in fulls]
        out_txt = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        i = 0
        with torch.no_grad():
            while i < len(order):
                j = i
                while j < len(order) and (j - i + 1) * (P + len(rows[order[j]]) + max_new) <= self.budget:
                    j += 1
                j = max(j, i + 1)
                idx = order[i:j]
                B, L = len(idx), max(len(rows[k]) for k in idx)
                ids = torch.full((B, L), self.tok.pad_token_id or 0, device=self.dev)
                sm = torch.zeros((B, L), dtype=torch.long, device=self.dev)
                for r, k in enumerate(idx):
                    ids[r, L - len(rows[k]):] = torch.tensor(rows[k], device=self.dev)
                    sm[r, L - len(rows[k]):] = 1
                mask = torch.cat([torch.ones((B, P), dtype=torch.long, device=self.dev), sm], 1)
                pos = P + torch.cumsum(sm, 1) - 1
                pos = pos.clamp(min=P)
                cache = self._prefix(pre)
                cache.batch_repeat_interleave(B)
                gen = [[] for _ in idx]
                done = [False] * B
                last = pos[:, -1]
                cur_ids, cur_pos = ids, pos
                for step in range(max_new):
                    out = self.model(input_ids=cur_ids, attention_mask=mask, position_ids=cur_pos,
                                     past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    nxt = out.logits[:, -1].argmax(-1)
                    for r in range(B):
                        if not done[r]:
                            gen[r].append(nxt[r].item())
                            if stop in self.tok.decode(gen[r]) or nxt[r].item() == self.tok.eos_token_id:
                                done[r] = True
                    if all(done):
                        break
                    last = last + 1
                    cur_ids, cur_pos = nxt.unsqueeze(1), last.unsqueeze(1)
                    mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device=self.dev)], 1)
                for r, k in enumerate(idx):
                    t = self.tok.decode(gen[r], skip_special_tokens=True)
                    out_txt[k] = t.split(stop)[0] if stop and stop in t else t
                i = j
        return out_txt
