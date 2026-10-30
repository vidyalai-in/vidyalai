"""lab/safety.py: helpers for Interpretability Part 11 (interpretability for safety).

Owner: Interpretability Part 11 (writer p11). Other parts may import it; nobody else edits it.

load_gemma()        Gemma 2 2B base or -it in bf16 via plain transformers (no weight processing).
chat()              wrap a user message in Gemma's chat template, with an optional prefill of the model turn.
city_items()        Part 3's 567 GeoNames cities, each with its true country and one false country.
last_token_states() residual stream at the LAST token of each prompt, every layer, plus the logits there.
                    Prompts are grouped by token count, so batches need no padding.
ab_margin()         logit("A") - logit("B") at the last position: the model's lean between two lettered options.
cv_scores()         city-grouped k-fold logistic-regression probe; returns held-out scores per item.
auroc_ci()          AUROC with a bootstrap CI that resamples groups (cities).
sae_encode()        JumpReLU SAE encode/decode in fp32 for a batch of activations (Gemma Scope weights).

Conventions
- "Layer l" is hidden_states[l] from Hugging Face: l = 0 is the (scaled) embedding, l = k is the output
  of decoder layer k-1. Gemma Scope's layer_12 SAE reads the output of decoder layer 12 = hidden_states[13].
- Tested with torch 2.11, transformers 5.21.1, scikit-learn 1.9.1, on an RTX 4060 Laptop GPU (8 GB).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

BASE = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16"
IT = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-it-bf16"
CITIES = "/home/amit/.cache/vidyalai-scratch/p03/runs/truth_data.json"


def load_gemma(path=IT, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
    return model, tok


def chat(user: str, prefill: str = "") -> str:
    """Gemma 2 chat format (the tokenizer adds <bos>)."""
    return f"<start_of_turn>user\n{user.strip()}<end_of_turn>\n<start_of_turn>model\n{prefill}"


def city_items(path=CITIES):
    """[{city, true, false}] for the 567 cities of Part 3 (GeoNames, CC BY 4.0)."""
    rows = json.load(open(path))
    out = {}
    for r in rows:
        d = out.setdefault(r["city"], {"city": r["city"], "true": r["true_country"]})
        if r["label"] == 0:
            d["false"] = r["country"]
    return list(out.values())


@torch.no_grad()
def last_token_states(model, tok, prompts, layers=None, batch=32, keep_ids=None):
    """Returns (H [N, len(layers), d] float16 numpy, logits_last [N, vocab] float32 torch on CPU).
    With keep_ids, logits_last holds only those token columns plus a last column with the argmax id
    (copying full 256k-token logits to the CPU is slow). No padding: prompts are grouped by token length."""
    ids = [tok(p, return_tensors=None)["input_ids"] for p in prompts]
    groups = defaultdict(list)
    for i, t in enumerate(ids):
        groups[len(t)].append(i)
    n_layers = model.config.num_hidden_layers + 1
    layers = list(range(n_layers)) if layers is None else list(layers)
    H = np.zeros((len(prompts), len(layers), model.config.hidden_size), dtype=np.float16)
    keep = {}
    dev = next(model.parameters()).device
    for n, idx in groups.items():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            x = torch.tensor([ids[i] for i in b], device=dev)
            out = model(input_ids=x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            hs = torch.stack([out.hidden_states[l][:, -1] for l in layers], 1)
            H[b] = hs.float().cpu().numpy().astype(np.float16)
            last = out.logits[:, -1].float()
            if keep_ids is not None:
                last = torch.cat([last[:, keep_ids], last.argmax(-1, keepdim=True).float()], 1)
            last = last.cpu()
            for j, i in enumerate(b):
                keep[i] = last[j]
    return H, torch.stack([keep[i] for i in range(len(prompts))])


def letter_ids(tok):
    return tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")


def ab_margin(tok, logits):
    """logits: full rows, or rows kept with keep_ids=list(letter_ids(tok))."""
    if logits.shape[1] == 3:
        return (logits[:, 0] - logits[:, 1]).numpy()
    a, b = letter_ids(tok)
    return (logits[:, a] - logits[:, b]).numpy()


def cv_scores(X, y, groups, n_folds=5, C=1.0, seed=0, train_mask=None):
    """Held-out probe scores (decision function) with folds split by group. Features are standardized
    with training statistics. train_mask restricts which items may be used for training."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % n_folds))
    f = np.array([fold_of[g] for g in groups])
    s = np.zeros(len(y))
    tm = np.ones(len(y), bool) if train_mask is None else train_mask
    for k in range(n_folds):
        tr, te = (f != k) & tm, f == k
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def auroc_ci(y, s, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    est, bs = auroc(y, s), []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[idx])) == 2:
            bs.append(auroc(y[idx], s[idx]))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


@torch.no_grad()
def sae_encode(sae, x):
    """x [T, d] any dtype -> (f [T, n] fp32, x_hat [T, d] fp32) with a Gemma Scope JumpReLU SAE."""
    x = x.float()
    pre = x @ sae.W_enc + sae.b_enc
    f = pre * (pre > sae.threshold)
    return f, f @ sae.W_dec + sae.b_dec
