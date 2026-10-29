"""lab/probes.py: activation collection to disk, linear and MLP probes with held-out eval,
control tasks (Hewitt & Liang 2019) and bootstrap confidence intervals.

Owner: Interpretability Part 3 (linear probes). Other parts import it; nobody else edits it.

Conventions
- Models load with TransformerLens `HookedTransformer.from_pretrained_no_processing`, so no
  LayerNorm folding and no weight centering: the residual stream is the model's own hidden state
  (the same numbers as Hugging Face `output_hidden_states`).
- "Layer l" means the residual stream entering block l (`blocks.{l}.hook_resid_pre`), so layer 0
  is token + position embedding and layer n_layers is the output of the last block
  (`blocks.{n-1}.hook_resid_post`, before the final LayerNorm).
- Activations are stored as float16 memmaps, one file per layer: <dir>/layer{l:02d}.npy, shape [N, d_model].
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- model + tokens

def load_model(name: str, dtype=torch.float32, device: str = "cuda"):
    import transformer_lens as tl
    model = tl.HookedTransformer.from_pretrained_no_processing(name, device=device, dtype=dtype)
    model.eval()
    return model


def bos_id(tokenizer) -> int:
    """GPT-2 and Qwen2.5 both use <|endoftext|> as the document separator; we prepend it."""
    return tokenizer.convert_tokens_to_ids("<|endoftext|>")


def tokenize_with_spans(tokenizer, text: str, spans: list[tuple[int, int]]):
    """Tokenize `text` (BOS prepended) and return (token_ids, positions), where positions[i] is the
    index of the LAST token that starts inside character span i (the word's final subword)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = []
    for s, e in spans:
        cand = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
        pos.append(cand[-1] + 1 if cand else -1)  # +1 for the prepended BOS
    return [bos_id(tokenizer)] + ids, pos


# --------------------------------------------------------------------------- collection

@torch.no_grad()
def collect_activations(model, token_lists, position_lists, out_dir, batch_size=32, meta=None):
    """Run `model` on each token list and save the residual stream at the given positions for
    every layer 0..n_layers to float16 memmaps. Sequences are right-padded: in a causal model
    with absolute or rotary positions counted from 0, padding after a token never changes it."""
    os.makedirs(out_dir, exist_ok=True)
    L, d = model.cfg.n_layers, model.cfg.d_model
    N = sum(len(p) for p in position_lists)
    mms = [np.lib.format.open_memmap(f"{out_dir}/layer{l:02d}.npy", mode="w+", dtype=np.float16, shape=(N, d))
           for l in range(L + 1)]
    names = [f"blocks.{l}.hook_resid_pre" for l in range(L)] + [f"blocks.{L-1}.hook_resid_post"]
    row, maxabs = 0, 0.0
    order = np.argsort([len(t) for t in token_lists])  # length-sorted batches: less padding
    rows_of = {}
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        T = max(len(token_lists[i]) for i in idx)
        toks = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            toks[j, :len(token_lists[i])] = torch.tensor(token_lists[i])
        _, cache = model.run_with_cache(toks.to(model.cfg.device), names_filter=lambda n: n in names)
        bj, bp = [], []
        for j, i in enumerate(idx):
            for p in position_lists[i]:
                bj.append(j); bp.append(p)
            rows_of[int(i)] = (row, row + len(position_lists[i]))
            row += len(position_lists[i])
        bj, bp = torch.tensor(bj), torch.tensor(bp)
        start = row - len(bj)
        for l, n in enumerate(names):
            a = cache[n][bj, bp].float()
            maxabs = max(maxabs, a.abs().max().item())
            mms[l][start:row] = a.cpu().numpy().astype(np.float16)
        del cache
    for m in mms:
        m.flush()
    # rows are in length-sorted order; store the map back to the caller's item order
    item_rows = np.concatenate([np.arange(*rows_of[i]) for i in range(len(token_lists))])
    np.save(f"{out_dir}/item_rows.npy", item_rows)
    info = {"n_rows": N, "d_model": d, "n_layers": L, "max_abs": maxabs, "fp16_safe": maxabs < 6e4}
    info.update(meta or {})
    json.dump(info, open(f"{out_dir}/info.json", "w"), indent=1)
    return info


def load_layer(act_dir: str, layer: int) -> np.ndarray:
    """Rows in the caller's original item order (the order position_lists was given in)."""
    X = np.load(f"{act_dir}/layer{layer:02d}.npy", mmap_mode="r")
    return np.asarray(X[np.load(f"{act_dir}/item_rows.npy")], dtype=np.float32)


# --------------------------------------------------------------------------- probes

def _mlp(d_in, n_out, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def train_probe(Xtr, ytr, Xte, yte, kind="linear", n_classes=None, hidden=1024, epochs=10,
                lr=1e-3, weight_decay=1e-4, batch_size=512, seed=0, device="cuda"):
    """Standardize with train statistics, train a softmax probe (linear: one matrix + bias;
    mlp: one ReLU hidden layer) with Adam and cross-entropy, return test predictions."""
    torch.manual_seed(seed)
    n_classes = n_classes or int(max(ytr.max(), yte.max()) + 1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    Xte_t = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    d = Xtr.shape[1]
    net = (nn.Linear(d, n_classes) if kind == "linear" else _mlp(d, n_classes, hidden)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr_t), device=device, generator=g)
        for b in range(0, len(perm), batch_size):
            i = perm[b:b + batch_size]
            loss = nn.functional.cross_entropy(net(Xtr_t[i]), ytr_t[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        pred_te = torch.cat([net(Xte_t[b:b + 8192]).argmax(-1) for b in range(0, len(Xte_t), 8192)]).cpu().numpy()
        pred_tr = torch.cat([net(Xtr_t[b:b + 8192]).argmax(-1) for b in range(0, len(Xtr_t), 8192)]).cpu().numpy()
    out = {"pred": pred_te, "acc": float((pred_te == yte).mean()), "train_acc": float((pred_tr == ytr).mean())}
    if kind == "linear":
        W = net.weight.detach().cpu().numpy() / sd  # direction in raw activation space
        out["W_raw"] = W
    return out


def mass_mean_direction(X, y):
    """Difference of class means (true minus false): Marks & Tegmark's mass-mean probe direction."""
    return X[y == 1].mean(0) - X[y == 0].mean(0)


# --------------------------------------------------------------------------- control tasks + stats

def control_labels(word_types, train_labels, seed=0):
    """Hewitt & Liang (2019) control task: each word TYPE gets one label, drawn independently from
    the empirical label distribution of the training tokens. Returns {type: label}."""
    rng = np.random.default_rng(seed)
    vals, counts = np.unique(train_labels, return_counts=True)
    types = sorted(set(word_types))
    draw = rng.choice(vals, size=len(types), p=counts / counts.sum())
    return dict(zip(types, draw.tolist()))


def bootstrap_ci(correct, groups=None, n_boot=1000, seed=0, alpha=0.05):
    """Mean of `correct` (0/1 or float per item) with a percentile bootstrap CI that resamples
    whole groups (sentences, cities) when `groups` is given."""
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(correct))
    uniq, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=correct, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(uniq), len(uniq))
        boots.append(s[k].sum() / c[k].sum())
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(correct.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a, b, groups, n_boot=1000, seed=0):
    """CI for mean(a) - mean(b) when a and b are scored on the same items (e.g. selectivity)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), groups, n_boot, seed)
