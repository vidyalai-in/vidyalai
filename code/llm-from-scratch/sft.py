"""Supervised fine-tuning (Build an LLM From Scratch, Part 10).

    python sft.py prepare                                   # tokenize smol-smoltalk once (CPU)
    python sft.py train --out runs/sft --lr 1e-4 --epochs 2 --compile
    python sft.py train --out runs/sft-all --train-on all ...   # the all-tokens ablation

The chat template uses the special tokens Part 2 reserved. Loss is computed on the assistant's
tokens only (targets of -1 elsewhere, Part 4's ignore_index), and normalized by the step's total
count of real targets (Part 5's trap). Conversations are padded, never packed: each micro-batch
holds conversations of similar length, padded on the right.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

EOT, SYSTEM, USER, ASSISTANT, END = 0, 1, 2, 3, 4          # Part 2's reserved ids
ROLE = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "smol-smoltalk")

# ---------------------------------------------------------------- the chat template


def render(messages, tok, train_on="assistant", bodies=None):
    """One conversation -> (ids, targets). ids start with <|endoftext|>, like every pretraining
    document; each message is <|role|> content <|end|>. targets[i] is the token after ids[i] if
    that token is trained, else -1: the assistant's content and its <|end|>, or everything."""
    ids, learn = [EOT], [False]
    for k, m in enumerate(messages):
        body = bodies[k] if bodies else tok.encode(m["content"].strip()).ids
        ids += [ROLE[m["role"]]] + body + [END]
        trained = train_on == "all" or m["role"] == "assistant"
        learn += [train_on == "all"] + [trained] * (len(body) + 1)
    targets = [t if ok else -1 for t, ok in zip(ids[1:], learn[1:])] + [-1]
    return ids, targets


def prompt_ids(messages, tok):
    """The conversation so far, then <|assistant|>: the model writes until it emits <|end|>."""
    return render(messages, tok)[0] + [ASSISTANT]


def chat_tokenizer():
    from tokenizer import load_production
    tok = load_production()
    tok.encode_special_tokens = True        # a literal "<|end|>" inside a message is just text
    return tok

# ---------------------------------------------------------------- data


def prepare(max_len=1024, n_val=2000, drop=("self-oss-instruct", "longalign")):
    """Tokenize smol-smoltalk's train and test splits into flat arrays: ids, an is-target mask
    (assistant tokens) and offsets. Drops code and long-document sources, conversations over
    max_len input tokens even when cut
    to whole exchanges, exact duplicates, and train conversations whose first user turn is in
    validation. Validation keeps the first n_val conversations of the test split that pass."""
    import glob
    import hashlib
    from array import array
    import pyarrow.parquet as pq
    tok, stats, val_prompts = chat_tokenizer(), {}, set()
    def first_user(r):
        return next(m["content"].strip() for m in r["messages"] if m["role"] == "user")
    for split, pattern in (("val", "test-*"), ("train", "train-*")):
        s = {"rows": 0, "dropped_source": 0, "duplicate": 0, "in_val": 0, "too_long": 0, "truncated": 0}
        seen, ids, mask, offsets, sources = set(), array("H"), bytearray(), [0], []
        for rows in (b.to_pylist() for f in sorted(glob.glob(f"{DATA}/data/{pattern}.parquet"))
                     for b in pq.ParquetFile(f).iter_batches(batch_size=5000)):
            flat = tok.encode_batch([m["content"].strip() for r in rows for m in r["messages"]])
            starts = np.cumsum([0] + [len(r["messages"]) for r in rows])
            for n, r in enumerate(rows):
                s["rows"] += 1
                key = hashlib.md5(json.dumps(r["messages"], sort_keys=True).encode()).digest()
                if r["source"] in drop:
                    s["dropped_source"] += 1; continue
                if key in seen:
                    s["duplicate"] += 1; continue
                if split == "train" and first_user(r) in val_prompts:
                    s["in_val"] += 1; continue
                bodies = [e.ids for e in flat[starts[n]:starts[n + 1]]]
                x, y = fit(r["messages"], bodies, max_len)
                if x is None:
                    s["too_long"] += 1; continue
                s["truncated"] += len(y) < sum(len(b) + 2 for b in bodies) + 1
                if split == "val" and len(sources) == n_val:
                    continue                          # keep counting rows, stop keeping
                seen.add(key)
                if split == "val":
                    val_prompts.add(first_user(r))
                ids.extend(x); mask.extend(bytes(t != -1 for t in y))
                offsets.append(len(ids)); sources.append(r["source"])
        s["kept"] = len(sources); stats[split] = s
        np.save(f"{DATA}/{split}_ids.npy", np.frombuffer(ids, dtype=np.uint16))
        np.save(f"{DATA}/{split}_mask.npy", np.frombuffer(bytes(mask), dtype=bool))
        np.save(f"{DATA}/{split}_offsets.npy", np.array(offsets, dtype=np.int64))
        json.dump(sources, open(f"{DATA}/{split}_sources.json", "w"))
    json.dump(stats, open(f"{DATA}/prepare_stats.json", "w"), indent=1)
    return stats


def fit(messages, bodies, max_len):
    """The conversation, or its longest prefix of whole exchanges (ending with an assistant turn)
    that fits in max_len input tokens; (None, None) if even the first exchange is too long."""
    ends = [i + 1 for i, m in enumerate(messages) if m["role"] == "assistant"]
    for end in reversed(ends):
        x, y = render(messages[:end], None, bodies=bodies[:end])
        if len(x) - 1 <= max_len:
            return x, y
    return None, None


class Conversations:
    """A tokenized split. example(i, train_on) -> (ids, targets) as numpy arrays."""

    def __init__(self, split, limit=None):
        self.ids = np.load(f"{DATA}/{split}_ids.npy")
        self.mask = np.load(f"{DATA}/{split}_mask.npy")
        self.off = np.load(f"{DATA}/{split}_offsets.npy")
        self.n = len(self.off) - 1 if limit is None else min(limit, len(self.off) - 1)
        self.lengths = np.diff(self.off)[:self.n] - 1            # input tokens per conversation

    def example(self, i, train_on="assistant"):
        a, b = self.off[i], self.off[i + 1]
        ids = self.ids[a:b].astype(np.int64)
        trained = np.ones(b - a - 1, bool) if train_on == "all" else self.mask[a:b - 1]
        return ids[:-1], np.where(trained, ids[1:], -1)     # mask[k]: is ids[k + 1] a target?


def plan(data, n_epochs, budget, seed, pad_to=64):
    """Every epoch's micro-batches as (conversation indices, padded length). Each epoch shuffles,
    sorts windows of 1,000 conversations by length, cuts them into groups of at most `budget`
    padded tokens, and shuffles the groups. Padding goes to a multiple of `pad_to`."""
    epochs = []
    for e in range(n_epochs):
        rng = random.Random(seed + e)
        order = list(range(data.n))
        rng.shuffle(order)
        groups = []
        for w in range(0, len(order), 1000):
            window = sorted(order[w:w + 1000], key=lambda i: data.lengths[i])
            i = 0
            while i < len(window):
                j = i + 1
                while j < len(window) and (j - i + 1) * _pad(data.lengths[window[j]], pad_to) <= budget:
                    j += 1
                groups.append((window[i:j], _pad(data.lengths[window[j - 1]], pad_to)))
                i = j
        rng.shuffle(groups)
        epochs.append(groups)
    return epochs


def _pad(n, m):
    return -(-int(n) // m) * m


def collate(data, group, T, train_on):
    """Right-padded inputs and targets. Padding is <|endoftext|> in x and -1 in y: causal attention
    means it never affects the real tokens before it, and it is never trained on."""
    x = torch.zeros(len(group), T, dtype=torch.long)
    y = torch.full((len(group), T), -1, dtype=torch.long)
    for r, k in enumerate(group):
        ids, tg = data.example(k, train_on)
        x[r, :len(ids)], y[r, :len(tg)] = torch.from_numpy(ids), torch.from_numpy(tg)
    return x, y

# ---------------------------------------------------------------- evaluation and chat


@torch.no_grad()
def response_loss(model, data, budget=8192):
    """Held-out loss on the assistant's tokens only: all of them, the content without <|end|>,
    and <|end|> alone. Sums of nats over every validation conversation, then divided."""
    model.eval()
    sums = {"all": [0.0, 0], "content": [0.0, 0], "end": [0.0, 0]}
    for group, T in plan(data, 1, budget, seed=0)[0]:
        x, y = (t.cuda() for t in collate(data, group, T, "assistant"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        nll = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                                ignore_index=-1, reduction="none").view_as(y)
        for name, keep in (("all", y != -1), ("content", (y != -1) & (y != END)), ("end", y == END)):
            sums[name][0] += nll[keep].sum().item()
            sums[name][1] += int(keep.sum())
    model.train()
    return {k: s / n for k, (s, n) in sums.items()} | {"targets": sums["all"][1]}


@torch.no_grad()
def chat(model, tok, messages, max_new=256, temperature=0.0, top_k=None, seed=0):
    """The model's reply to a conversation that ends with a user turn. Returns the text, whether it
    ended with <|end|> within max_new tokens, and how many tokens it wrote."""
    idx = torch.tensor([prompt_ids(messages, tok)], device="cuda")
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new, temperature=temperature, top_k=top_k, eos_id=END)
    new = out[0, idx.shape[1]:].tolist()
    ended = bool(new) and new[-1] == END
    return tok.decode(new[:-1] if ended else new), ended, len(new)

# ---------------------------------------------------------------- training


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "train"])
    p.add_argument("--out", default="runs/sft")
    p.add_argument("--init", default=os.path.join(HERE, "..", "t07", "runs", "main88", "model_03360.pt"))
    p.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    p.add_argument("--limit", type=int, default=None, help="use only the first N training conversations")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--budget", type=int, default=16384, help="padded tokens per micro-batch")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--val-limit", type=int, default=500, help="validation conversations scored during the run")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--stop-at", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fused", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def train(a):
    from snapshots import load
    from train import lr_at, make_optimizer               # Part 5's schedule and optimizer groups
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    data, val = Conversations("train", a.limit), Conversations("val", a.val_limit)
    epochs = plan(data, a.epochs, a.budget, a.seed)
    steps = [[g[k:k + a.accum] for k in range(0, len(g), a.accum)] for g in epochs]
    steps = [s for e in steps for s in e]                  # every optimizer step's micro-batches
    a.steps = len(steps)
    raw = load(a.init).train()                             # FP32 weights from Part 7
    opt = make_optimizer(raw, a)
    step, ckpt_path = 0, os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step = ck["step"]
    model = torch.compile(raw, dynamic=True) if a.compile else raw
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def write(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    write({"event": "resume" if step else "start", "step": step, "args": vars(a),
           "conversations": data.n, "torch": torch.__version__})
    stop, t0 = a.stop_at or a.steps, time.time()
    while step < stop:
        if a.val_every and step % a.val_every == 0:
            write({"step": step, "val": response_loss(model, val)})
        t = time.time()
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        micro = [collate(data, grp, T, a.train_on) for grp, T in steps[step]]
        n_step = sum(int((y != -1).sum()) for _, y in micro)      # real targets in this whole step
        loss_sum = torch.zeros((), device="cuda")
        for x, y in micro:
            n = int((y != -1).sum())
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y) * (n / n_step)    # this micro-batch's share of the step's mean
            loss.backward()
            loss_sum += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val, dt = loss_sum.item(), time.time() - t
        padded = sum(x.numel() for x, _ in micro)
        real = sum(int((x != 0).sum()) + x.shape[0] for x, _ in micro)   # + the <|endoftext|> start
        step += 1
        write({"step": step - 1, "loss": loss_val, "lr": lr, "grad_norm": gn.item(), "targets": n_step,
               "tokens": real, "padded": padded, "dt": round(dt, 4), "tok_per_s": round(padded / dt),
               "elapsed": round(time.time() - t0, 2)})
        if step % a.ckpt_every == 0 or step == stop:
            torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    if step == a.steps:
        torch.save(raw.state_dict(), os.path.join(a.out, "model_final.pt"))   # weights only
    write({"step": step, "val": response_loss(model, val)})
    write({"event": "end", "step": step, "elapsed": round(time.time() - t0, 2),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})


if __name__ == "__main__":
    args = get_args()
    print(json.dumps(prepare(), indent=1)) if args.cmd == "prepare" else train(args)
