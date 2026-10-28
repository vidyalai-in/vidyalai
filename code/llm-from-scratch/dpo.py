"""Preference tuning with DPO (Build an LLM From Scratch, Part 11).

    python dpo.py prepare                                   # UltraFeedback pairs -> token arrays (CPU)
    python dpo.py reference --init <sft weights>            # the frozen model's log-probs, once
    python dpo.py train --out runs/dpo --beta 0.1 --lr 1e-6 --init <sft weights>

A pair is one prompt and two replies, chosen and rejected, each laid out with Part 10's chat
template and scored on the reply only (its tokens and <|end|>). The loss is Rafailov et al.'s:
-log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]).
The reference is the SFT model, frozen, so its log-probabilities of the fixed pairs never change:
they are computed once and cached, and only the policy is in memory during training.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from sft import END, chat_tokenizer, plan, prompt_ids          # Part 10's template and batching

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- pairs


def encode_pair(prompt, chosen, rejected, tok):
    """Both replies after the same prompt: (chosen ids, rejected ids, index of the first reply token)."""
    p = prompt_ids([{"role": "user", "content": prompt}], tok)    # <|endoftext|><|user|>...<|end|><|assistant|>
    return (p + tok.encode(chosen.strip()).ids + [END],
            p + tok.encode(rejected.strip()).ids + [END], len(p))


def save_pairs(name, split, pairs):
    """pairs: list of (chosen ids, rejected ids, start). Sequence 2i is pair i's chosen reply, 2i + 1
    its rejected one; start[k] is where sequence k's reply begins."""
    d = os.path.join(HERE, "data", name)
    os.makedirs(d, exist_ok=True)
    seqs = [s for c, r, _ in pairs for s in (c, r)]
    np.save(f"{d}/{split}_ids.npy", np.array([t for s in seqs for t in s], dtype=np.uint16))
    np.save(f"{d}/{split}_offsets.npy", np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64))
    np.save(f"{d}/{split}_start.npy", np.array([st for *_, st in pairs for _ in (0, 1)], dtype=np.int64))


def prepare(max_len=1024, max_prompt=512, seed=0):
    """UltraFeedback binarized (train_prefs, test_prefs) -> data/ultrafeedback/{train,test}_*.npy.
    Drops ties, empty or identical replies, anything containing '{' (code, JSON, LaTeX: a byte the
    base model never saw), prompts over max_prompt tokens, pairs where either reply does not fit in
    max_len input tokens, and training prompts that also appear in the test split. Shuffles."""
    import pyarrow.parquet as pq
    tok, stats, test_prompts = chat_tokenizer(), {}, set()
    src = os.path.join(HERE, "data", "ultrafeedback_binarized")
    for split in ("test", "train"):
        rows = pq.read_table(f"{src}/{split}_prefs.parquet").to_pylist()
        s = dict.fromkeys(["rows", "tie", "empty_or_identical", "brace", "in_test", "prompt_too_long",
                           "reply_too_long", "kept"], 0)
        pairs = []
        for r in rows:
            s["rows"] += 1
            prompt, c, j = r["prompt"].strip(), r["chosen"][-1]["content"], r["rejected"][-1]["content"]
            if r["score_chosen"] <= r["score_rejected"]:
                s["tie"] += 1; continue
            if not c.strip() or not j.strip() or c.strip() == j.strip():
                s["empty_or_identical"] += 1; continue
            if "{" in prompt + c + j:
                s["brace"] += 1; continue
            if split == "train" and prompt in test_prompts:
                s["in_test"] += 1; continue
            cid, rid, start = encode_pair(prompt, c, j, tok)
            if start > max_prompt:
                s["prompt_too_long"] += 1; continue
            if max(len(cid), len(rid)) - 1 > max_len:
                s["reply_too_long"] += 1; continue
            pairs.append((cid, rid, start))
            if split == "test":
                test_prompts.add(prompt)
        pairs = [pairs[i] for i in np.random.default_rng(seed).permutation(len(pairs))]
        s["kept"] = len(pairs)
        save_pairs("ultrafeedback", split, pairs)
        stats[split] = s
    json.dump(stats, open(os.path.join(HERE, "data", "ultrafeedback", "prepare_stats.json"), "w"), indent=1)
    return stats


class Pairs:
    """A tokenized split of preference pairs. n pairs; lengths[i] = the longer reply's input length."""

    def __init__(self, name, split, limit=None):
        d = os.path.join(HERE, "data", name)
        self.prefix = f"{d}/{split}"
        self.ids = np.load(f"{self.prefix}_ids.npy")
        self.off = np.load(f"{self.prefix}_offsets.npy")
        self.start = np.load(f"{self.prefix}_start.npy")
        total = (len(self.off) - 1) // 2
        self.n = total if limit is None else min(limit, total)
        seq_len = np.diff(self.off) - 1
        self.lengths = np.maximum(seq_len[0::2], seq_len[1::2])[:self.n]

    def seq(self, k):
        return self.ids[self.off[k]:self.off[k + 1]].astype(np.int64)

    def reference(self, init):
        """The cached reference log-probs, (n, 2) for chosen and rejected; made by `dpo.py reference`."""
        meta = json.load(open(f"{self.prefix}_ref.json"))
        assert os.path.realpath(meta["init"]) == os.path.realpath(init), "reference made from other weights"
        assert meta["pairs"] >= self.n, "reference covers fewer pairs: rerun dpo.py reference --limit"
        return torch.from_numpy(np.load(f"{self.prefix}_ref.npy")[:self.n])


def collate_pairs(data, group, T):
    """Chosen replies in the first len(group) rows, rejected in the rest, right-padded to T. Targets
    are -1 except on the reply's tokens and its <|end|>, as in Part 10."""
    B = len(group)
    x = torch.zeros(2 * B, T, dtype=torch.long)
    y = torch.full((2 * B, T), -1, dtype=torch.long)
    for r, i in enumerate(group):
        for row, k in ((r, 2 * i), (B + r, 2 * i + 1)):
            s, st = data.seq(k), data.start[k]
            x[row, :len(s) - 1] = torch.from_numpy(s[:-1])
            y[row, st - 1:len(s) - 1] = torch.from_numpy(s[st:])   # position t predicts token t + 1
    return x, y

# ---------------------------------------------------------------- the loss


def sequence_logps(model, x, y):
    """log pi(reply | prompt) for every row: the sum of the reply tokens' log-probabilities."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    logp = -F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-1,
                            reduction="none").view_as(y)
    return logp.sum(-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """Per-pair DPO loss, and the implicit rewards beta * log(pi / ref) of both replies."""
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss, chosen_reward.detach(), rejected_reward.detach()


@torch.no_grad()
def all_logps(model, data, budget=8192):
    """log pi of every pair's two replies, (n, 2), and their reply lengths in tokens."""
    was_training = model.training
    model.eval()
    out, n_tok = torch.zeros(data.n, 2, dtype=torch.float64), torch.zeros(data.n, 2, dtype=torch.long)
    for group, T in plan(data, 1, budget // 2, seed=0)[0]:
        x, y = collate_pairs(data, group, T)
        lp = sequence_logps(model, x.cuda(), y.cuda()).double().cpu()
        B = len(group)
        out[group, 0], out[group, 1] = lp[:B], lp[B:]
        n_tok[group, 0], n_tok[group, 1] = (y[:B] != -1).sum(1), (y[B:] != -1).sum(1)
    model.train(was_training)
    return out, n_tok


def evaluate_pairs(model, data, ref, beta):
    """Held-out DPO loss, preference accuracy (implicit reward of chosen > rejected), reward margin,
    and the mean change in log-probability of chosen and rejected replies, per reply and per token."""
    pi, n_tok = all_logps(model, data)
    loss, rc, rr = dpo_loss(pi[:, 0], pi[:, 1], ref[:, 0], ref[:, 1], beta)
    delta = pi - ref                                       # log pi - log ref, per reply
    return {"loss": loss.mean().item(), "acc": (rc > rr).double().mean().item(),
            "margin": (rc - rr).mean().item(), "chosen_reward": rc.mean().item(),
            "rejected_reward": rr.mean().item(),
            "chosen_logp": pi[:, 0].mean().item(), "rejected_logp": pi[:, 1].mean().item(),
            "chosen_dlogp_per_tok": (delta[:, 0].sum() / n_tok[:, 0].sum()).item(),
            "rejected_dlogp_per_tok": (delta[:, 1].sum() / n_tok[:, 1].sum()).item()}

# ---------------------------------------------------------------- the reference, once


def reference(a):
    """Score every pair with the frozen SFT model (FP32 weights, BF16 autocast, like training)."""
    from snapshots import load
    ref = load(a.init)
    for split in ("train", "test"):
        if not os.path.exists(os.path.join(HERE, "data", a.pairs, f"{split}_ids.npy")):
            continue
        data = Pairs(a.pairs, split, a.limit if split == "train" else None)
        lp, _ = all_logps(ref, data)
        np.save(f"{data.prefix}_ref.npy", lp.numpy())
        json.dump({"init": a.init, "pairs": data.n}, open(f"{data.prefix}_ref.json", "w"))
        print(split, data.n, "pairs scored")

# ---------------------------------------------------------------- training


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "reference", "train"])
    p.add_argument("--out", default="runs/dpo")
    p.add_argument("--init", default=os.path.join(HERE, "..", "t10", "runs", "sft88", "model_final.pt"))
    p.add_argument("--pairs", default="ultrafeedback", help="a folder under data/ made by prepare")
    p.add_argument("--limit", type=int, default=None, help="use only the first N training pairs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--budget", type=int, default=4096, help="padded tokens per micro-batch, both replies")
    p.add_argument("--accum", type=int, default=16, help="micro-batches per optimizer step")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-limit", type=int, default=500, help="held-out pairs scored during the run")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--stop-at", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--fused", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def train(a):
    from snapshots import load
    from train import lr_at, make_optimizer                # Part 5's schedule and optimizer groups
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    data, val = Pairs(a.pairs, "train", a.limit), Pairs("ultrafeedback", "test", a.val_limit)
    ref, ref_val = data.reference(a.init), val.reference(a.init)
    epochs = plan(data, a.epochs, a.budget // 2, a.seed)   # budget // 2: each pair is two rows
    steps = [[g[k:k + a.accum] for k in range(0, len(g), a.accum)] for g in epochs]
    steps = [s for e in steps for s in e]
    a.steps = len(steps)
    model = load(a.init).train()                           # the policy starts as the reference
    opt = make_optimizer(model, a)
    step, ckpt_path = 0, os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step = ck["step"]
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def write(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    write({"event": "resume" if step else "start", "step": step, "args": vars(a), "pairs": data.n,
           "torch": torch.__version__})
    stop, t0 = a.stop_at or a.steps, time.time()
    while step < stop:
        if a.val_every and step % a.val_every == 0:
            write({"step": step, "val": evaluate_pairs(model, val, ref_val, a.beta)})
        t = time.time()
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        n_step = sum(len(grp) for grp, _ in steps[step])   # pairs in this optimizer step
        stats = torch.zeros(6, dtype=torch.float64)
        tokens = 0
        for grp, T in steps[step]:
            x, y = collate_pairs(data, grp, T)
            tokens += x.numel()
            logp = sequence_logps(model, x.cuda(non_blocking=True), y.cuda(non_blocking=True))
            B, r = len(grp), ref[grp].cuda()
            loss, rc, rr = dpo_loss(logp[:B], logp[B:], r[:, 0], r[:, 1], a.beta)
            (loss.sum() / n_step).backward()                # every pair in the step counts once
            stats += torch.stack([loss.detach().sum(), (rc > rr).sum(), (rc - rr).sum(), rc.sum(),
                                  logp[:B].detach().sum(), logp[B:].detach().sum()]).double().cpu()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        s = (stats / n_step).tolist()
        dt = time.time() - t
        step += 1
        write({"step": step - 1, "loss": s[0], "acc": s[1], "margin": s[2], "chosen_reward": s[3],
               "rejected_reward": s[3] - s[2], "chosen_logp": s[4], "rejected_logp": s[5], "lr": lr,
               "grad_norm": gn.item(), "pairs": n_step, "padded": tokens, "dt": round(dt, 4),
               "tok_per_s": round(tokens / dt), "elapsed": round(time.time() - t0, 2)})
        if step % a.ckpt_every == 0 or step == stop:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    if step == a.steps:
        torch.save(model.state_dict(), os.path.join(a.out, "model_final.pt"))   # weights only
    write({"step": step, "val": evaluate_pairs(model, val, ref_val, a.beta)})
    write({"event": "end", "step": step, "elapsed": round(time.time() - t0, 2),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})


if __name__ == "__main__":
    args = get_args()
    {"prepare": lambda a: print(json.dumps(prepare(), indent=1)), "reference": reference,
     "train": train}[args.cmd](args)
