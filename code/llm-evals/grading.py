"""Grading open-ended outputs (Measuring LLMs, Part 6).

Three families of graders, all scored against human labels:
  reference metrics  exact match, token F1 (SQuAD-style), BLEU, chrF, ROUGE, BERTScore
  rubric judges      an LLM that rates one output on anchored 1-5 scales (pointwise)
  pairwise judges    an LLM that picks the better of two outputs (MT-Bench "pair-v2" prompt)

Agreement statistics: summary-level and system-level correlation (Pearson, Spearman, Kendall
tau-b) with bootstrap intervals over documents, Cohen's kappa, and the MT-Bench agreement
definition for pairwise votes with and without ties (Zheng et al. 2023, arXiv 2306.05685).

API work goes through the project ledger, in Message Batches (50% off). The batch id is written
to disk right after create so results can be recovered if the poller dies. BERTScore needs torch
and transformers (the interp venv); everything else runs on numpy, scipy, sacrebleu and
rouge-score.
"""
import collections
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as _st

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")

# Thinking off (batch-1 lesson): Haiku 5.5 thinks when `thinking` is omitted; Sonnet 5.5 rejects
# {"type": "disabled"} and takes {"type": "between_tools"}, its no-thinking-before-reply setting.
THINK_OFF = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


# ---------------------------------------------------------------- reference metrics -------------

def normalize(text):
    """SQuAD v1.1 answer normalization: lowercase, drop punctuation and articles, squeeze spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    """Harmonic mean of token precision and recall after normalization (SQuAD v1.1)."""
    p, r = normalize(pred).split(), normalize(ref).split()
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


_ROUGE = None


def rouge(pred, refs, kinds=("rouge1", "rouge2", "rougeL")):
    """ROUGE F-measures (Google's rouge-score, Porter stemming). With several references,
    each kind takes the best score over references, as rouge-score's score_multi does."""
    global _ROUGE
    from rouge_score import rouge_scorer
    if _ROUGE is None:
        _ROUGE = rouge_scorer.RougeScorer(list(kinds), use_stemmer=True)
    refs = [refs] if isinstance(refs, str) else refs
    s = _ROUGE.score_multi(refs, pred)
    return {k: s[k].fmeasure for k in kinds}


def bleu(pred, refs):
    """Sentence BLEU (sacrebleu, exponential smoothing), 0-100."""
    import sacrebleu
    refs = [refs] if isinstance(refs, str) else refs
    return sacrebleu.sentence_bleu(pred, refs).score


def chrf(pred, refs):
    """Sentence chrF (character 6-gram F-score, beta = 2; sacrebleu), 0-100."""
    import sacrebleu
    refs = [refs] if isinstance(refs, str) else refs
    return sacrebleu.sentence_chrf(pred, refs).score


class BertScorer:
    """BERTScore F1 (Zhang et al. 2019, arXiv 1904.09675) without idf weighting or baseline
    rescaling: embed both texts with one hidden layer of a pretrained encoder, match every token
    to its most similar token on the other side (cosine), and take the harmonic mean of the
    average best match in each direction. Special tokens are left out, as in bert-score.
    Rescaling is linear, so it changes no correlation. Default: roberta-large, layer 17, the
    bert-score package's default for English."""

    def __init__(self, model="roberta-large", layer=17, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.layer, self.device = torch, layer, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()

    def embed(self, texts, batch_size=64):
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=510,
                           return_tensors="pt", return_special_tokens_mask=True).to(self.device)
            special = enc.pop("special_tokens_mask").bool()
            with torch.no_grad():
                h = self.model(**enc, output_hidden_states=True).hidden_states[self.layer].float()
            h = torch.nn.functional.normalize(h, dim=-1)
            keep = enc["attention_mask"].bool() & ~special
            out += [h[j][keep[j]].cpu() for j in range(len(h))]
        return out

    def f1(self, cands, refs):
        """cands: list of strings; refs: list of lists of strings. Max F1 over each item's refs."""
        flat = sorted({r for rs in refs for r in rs} | set(cands))
        emb = dict(zip(flat, self.embed(flat)))
        scores = []
        for c, rs in zip(cands, refs):
            best = -1.0
            for r in rs:
                sim = emb[c] @ emb[r].T
                p, rcl = sim.max(1).values.mean().item(), sim.max(0).values.mean().item()
                best = max(best, 2 * p * rcl / (p + rcl))
            scores.append(best)
        return scores


# ---------------------------------------------------------------- correlation with humans --------

CORR = {"pearson": lambda a, b: _st.pearsonr(a, b)[0],
        "spearman": lambda a, b: _st.spearmanr(a, b)[0],
        "kendall": lambda a, b: _st.kendalltau(a, b)[0]}  # tau-b, which handles ties


def per_doc(metric, human, method="kendall"):
    """One correlation per document (NaN where either row is constant)."""
    return np.array([CORR[method](m, h) if np.ptp(m) > 0 and np.ptp(h) > 0 else np.nan
                     for m, h in zip(np.asarray(metric, float), np.asarray(human, float))])


def summary_level(metric, human, method="kendall"):
    """metric, human: (docs, systems) arrays. Correlate across the systems' outputs for each
    document, then average over documents (constant rows are skipped)."""
    return float(np.nanmean(per_doc(metric, human, method)))


def system_level(metric, human, method="kendall"):
    """Correlate the systems' mean scores over all documents."""
    return float(CORR[method](np.mean(metric, 0), np.mean(human, 0)))


def boot_corr(metric, human, level_fn, method="kendall", n_boot=2000, seed=0):
    """Point estimate and percentile 95% interval, resampling documents with replacement."""
    rng = np.random.default_rng(seed)
    metric, human = np.asarray(metric, float), np.asarray(human, float)
    n = len(metric)
    idx = rng.integers(0, n, (n_boot, n))
    if level_fn is summary_level:  # resampling documents = resampling the per-document values
        r = per_doc(metric, human, method)
        vals = np.nanmean(r[idx], 1)
    else:
        vals = [level_fn(metric[i], human[i], method) for i in idx]
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return dict(est=level_fn(metric, human, method), lo=float(lo), hi=float(hi))


def boot_corr_diff(m1, m2, human, level_fn, method="kendall", n_boot=2000, seed=0):
    """Paired bootstrap of corr(m2, human) - corr(m1, human) over the same resampled documents."""
    rng = np.random.default_rng(seed)
    m1, m2, human = (np.asarray(x, float) for x in (m1, m2, human))
    n = len(human)
    idx = rng.integers(0, n, (n_boot, n))
    if level_fn is summary_level:
        r1, r2 = per_doc(m1, human, method), per_doc(m2, human, method)
        vals = np.nanmean(r2[idx], 1) - np.nanmean(r1[idx], 1)
    else:
        vals = [level_fn(m2[i], human[i], method) - level_fn(m1[i], human[i], method) for i in idx]
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return dict(est=level_fn(m2, human, method) - level_fn(m1, human, method), lo=float(lo), hi=float(hi))


# ---------------------------------------------------------------- agreement ----------------------

def cohen_kappa(a, b, labels=None):
    """(p_o - p_e) / (1 - p_e) for two raters' labels on the same items."""
    a, b = list(a), list(b)
    labels = labels or sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_stat(fn, n_items, groups=None, n_boot=2000, seed=0):
    """Percentile 95% interval of fn(idx) over resampled item indices. With ``groups`` (one label
    per item), whole groups are resampled, e.g. every vote on one question."""
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(n_boot):
            vals.append(fn(rng.integers(0, n_items, n_items)))
    else:
        g = np.asarray(groups)
        members = [np.flatnonzero(g == u) for u in np.unique(g)]
        for _ in range(n_boot):
            pick = rng.integers(0, len(members), len(members))
            vals.append(fn(np.concatenate([members[j] for j in pick])))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def vote_agreement(x, y, with_ties=True):
    """Share of vote pairs that agree, as in MT-Bench (Zheng et al. 2023, sec. 4.2): with ties
    ("S1") a tie only matches a tie; without ties ("S2") pairs where either side said tie are
    dropped. x, y: sequences of 'A', 'B' or 'tie' for the same pairs. Returns (agree, n)."""
    pairs = [(u, v) for u, v in zip(x, y) if with_ties or (u != "tie" and v != "tie")]
    return (sum(u == v for u, v in pairs) / len(pairs) if pairs else float("nan")), len(pairs)


# ---------------------------------------------------------------- judge prompts ------------------

# MT-Bench pairwise judge prompt "pair-v2" and its reference-guided variant "pair-math-v1",
# from lm-sys/FastChat fastchat/llm_judge/data/judge_prompts.jsonl (Apache-2.0).
PROMPTS = {json.loads(l)["name"]: json.loads(l) for l in
           open("/home/amit/.cache/vidyalai-scratch/e06/data/judge_prompts.jsonl")}


def pair_request(question, ans_a, ans_b, model="claude-haiku-5-5", reference=None, max_tokens=600):
    """Params for one pairwise verdict. Thinking off; the prompt asks for a short explanation
    and then "[[A]]", "[[B]]" or "[[C]]" (tie)."""
    p = PROMPTS["pair-math-v1" if reference else "pair-v2"]
    user = p["prompt_template"].format(question=question, answer_a=ans_a, answer_b=ans_b,
                                       ref_answer_1=reference or "")
    return {"model": model, "max_tokens": max_tokens, "system": p["system_prompt"],
            "thinking": THINK_OFF[model], "messages": [{"role": "user", "content": user}]}


def parse_verdict(text):
    """'A', 'B', 'tie', or None if no verdict tag (the last tag wins, as in FastChat)."""
    tags = re.findall(r"\[\[([ABC])\]\]", text or "")
    return {"A": "A", "B": "B", "C": "tie"}[tags[-1]] if tags else None


def single_request(question, answer, model="claude-haiku-5-5", max_tokens=600):
    """MT-Bench "single-v1": rate one answer 1-10, output "[[rating]]"."""
    p = PROMPTS["single-v1"]
    user = p["prompt_template"].format(question=question, answer=answer)
    return {"model": model, "max_tokens": max_tokens, "system": p["system_prompt"],
            "thinking": THINK_OFF[model], "messages": [{"role": "user", "content": user}]}


def parse_rating(text):
    m = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", text or "")
    return float(m[-1]) if m else None


DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")

# Anchored rubric written for this part from the SummEval annotation guidelines
# (Fabbri et al. 2021, arXiv 2007.12626, sec. 4.1): one definition per dimension, 1-5 scale.
RUBRIC = {
    "coherence": "The summary is well structured and well organized: it builds from sentence to "
                 "sentence into a coherent body of information about the topic, not a heap of related "
                 "facts. 1 = disjointed or contradictory sentences; 3 = understandable but jumps around "
                 "or repeats; 5 = reads as one organized short text.",
    "consistency": "Every fact in the summary is supported by the article (no hallucinated or "
                   "contradicted facts, names, numbers or relations). 1 = several unsupported or wrong "
                   "claims; 3 = one unsupported or distorted claim; 5 = everything is supported.",
    "fluency": "Each sentence is well formed: grammatical, no fragments, no repeated words, no "
               "formatting debris, no missing capitals. 1 = many broken sentences; 3 = some errors that "
               "slow reading; 5 = no errors.",
    "relevance": "The summary keeps only the important information from the article and leaves out "
                 "redundancies and minor details. 1 = misses the main point; 3 = gets the main point but "
                 "misses key details or includes trivia; 5 = all key points, nothing minor.",
}

SCORE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": list(DIMENSIONS),
                "properties": {d: {"type": "integer"} for d in DIMENSIONS}}


def rubric_request(article, summary, model="claude-haiku-5-5", anchored=True, max_tokens=200):
    """Params for one pointwise 1-5 rating of a news summary on four dimensions. JSON output
    (structured outputs), thinking off. anchored=False gives only the dimension names."""
    if anchored:
        guide = "\n".join(f"- {d}: {RUBRIC[d]}" for d in DIMENSIONS)
    else:
        guide = "\n".join(f"- {d}" for d in DIMENSIONS)
    system = ("You grade machine-written summaries of news articles. Rate the summary on each "
              "dimension with an integer from 1 (worst) to 5 (best).\n" + guide +
              "\nReply with the JSON object only.")
    user = f"<article>\n{article}\n</article>\n\n<summary>\n{summary}\n</summary>"
    return {"model": model, "max_tokens": max_tokens, "system": system, "thinking": THINK_OFF[model],
            "output_config": {"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
            "messages": [{"role": "user", "content": user}]}


# Content-free padding for the verbosity test: says nothing about the question.
PADDING = ("\n\nI hope this answer is helpful. To summarize, the response above addresses the "
           "question that was asked and covers the main points in a clear and organized way. Each "
           "part of the answer was written with care so that it is easy to follow and understand. "
           "If any part of it is unclear, or if you would like more detail on a particular point, "
           "please feel free to ask, and I will be glad to explain further or to provide additional "
           "information, examples or clarification as needed.")


# ---------------------------------------------------------------- batches ------------------------

def run_batch(part, jobs, out_path, note, poll_s=20):
    """jobs: {custom_id: params}. Guards the worst-case cost, submits one Message Batch, writes the
    batch id to out_path + '.batch' at once, waits, records every succeeded result in the ledger
    and appends request + response lines to out_path. Returns a summary dict."""
    import ledger
    cl = ledger.client()
    est = sum(ledger.estimate_cost(p["model"], len(json.dumps(p)) // 2, p["max_tokens"])
              for p in jobs.values()) * 0.5
    ledger.guard(part, est)
    batch = cl.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p in jobs.items()])
    Path(str(out_path) + ".batch").write_text(json.dumps({"batch": batch.id, "note": note, "n": len(jobs)}))
    t0 = time.time()
    print(f"submitted {batch.id} ({len(jobs)} requests, est ${est:.3f})", flush=True)
    return collect(part, batch.id, jobs, out_path, note, poll_s, t0, est)


def collect(part, batch_id, jobs, out_path, note, poll_s=20, t0=None, est=None):
    """Wait for a batch to end and record its results (also used to recover a batch by id)."""
    import ledger
    cl = ledger.client()
    t0 = t0 or time.time()
    while cl.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll_s)
    cost, counts = 0.0, collections.Counter()
    with open(out_path, "a") as f:
        for it in cl.messages.batches.results(batch_id):
            p = jobs[it.custom_id]
            rec = {"id": it.custom_id, "batch": batch_id, "params": p, "type": it.result.type}
            counts[it.result.type] += 1
            if it.result.type == "succeeded":
                m = it.result.message
                cost += ledger.record(part, p["model"], m.usage, note=note, batch=True)
                rec.update(text="".join(c.text for c in m.content if c.type == "text"),
                           stop=m.stop_reason, usage=m.usage.model_dump())
                counts[f"stop:{m.stop_reason}"] += 1
            f.write(json.dumps(rec) + "\n")
    return dict(batch=batch_id, n=len(jobs), seconds=round(time.time() - t0), cost=cost,
                estimate=est, counts=dict(counts))


def load_runs(path):
    """{custom_id: record} from a JSONL run file (the last record per id wins)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def truecase(text):
    """Undo SummEval's lowercased, space-tokenized output in the cheap way: rejoin clitics and
    punctuation and capitalize each sentence's first letter (proper nouns stay lowercase)."""
    t = text.replace(" n't", "n't").replace(" 's", "'s").replace(" '", "'")
    t = re.sub(r" ([.,;:!?%)])", r"\1", t).replace("( ", "(").replace("`` ", '"').replace(" ''", '"')
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t.strip())
    return re.sub(r"\bi\b", "I", t)
