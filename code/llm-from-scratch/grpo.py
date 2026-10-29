"""RL with verifiable rewards: GRPO on addition (Build an LLM From Scratch, Part 12).

    python grpo.py warmstart --out runs/warm --n 4000 --lr 3e-4 --warmup 5 --fused   # SFT on the format
    python grpo.py warmstart ... --chat 4000 --batch 128        # mixed with Part 10's chat data
    python grpo.py train --init runs/warm/model_final.pt --out runs/grpo --lr 1e-5 --steps 250 --fused
    python grpo.py train ... --no-std --agg token-sum              # Dr. GRPO
    python grpo.py train ... --reward contains                     # a checker that can be gamed

The task: "What is 347 + 582?" with 1- to 3-digit numbers, in Part 10's chat template. The checker:
the reply's last sentence must be "The answer is 929." with the right number. GRPO samples G replies
per prompt with Part 11's KV-cache sampler, scores them, and uses advantage = (r - group mean) /
group std on every token of a reply, inside PPO's clipped ratio, plus an optional k3 KL penalty to
the frozen starting model. Rewards are checked by a program, so no reward model is needed.
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F

from sampling import generate_batch                          # Part 11's batched sampler
from sft import END, chat_tokenizer, prompt_ids, render     # Part 10's chat template

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- the task and the checker

QUESTIONS = ["What is {a} + {b}?", "Add {a} and {b}.", "Compute {a} + {b}."]


def number(rng, digits):
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)


def problem(rng, max_digits=3, digits=None):
    """(question, a, b): each number has 1 to max_digits digits, or exactly `digits`."""
    a = number(rng, digits or rng.randint(1, max_digits))
    b = number(rng, digits or rng.randint(1, max_digits))
    return rng.choice(QUESTIONS).format(a=a, b=b), a, b


def problems(n, seed, avoid=frozenset(), **kw):
    """n distinct problems from a fixed seed, none of whose (a, b) is in `avoid`."""
    rng, out, seen = random.Random(seed), [], set(avoid)
    while len(out) < n:
        q, a, b = problem(rng, **kw)
        if (a, b) not in seen:
            seen.add((a, b))
            out.append((q, a, b))
    return out


def test_set():
    """The fixed held-out problems: 500 in distribution (1 to 3 digits), 200 with 4 digits each,
    200 with 5 digits each. Training never sees an (a, b) from the first set."""
    return {"test": problems(500, seed=12), "4digit": problems(200, seed=13, digits=4),
            "5digit": problems(200, seed=14, digits=5)}


FINAL = re.compile(r"The answer is (\d+)\.$")


def reward(reply, answer, kind="strict"):
    """1.0 if the reply is right, else 0.0. strict: the reply ends with "The answer is <answer>."
    contains: the answer appears anywhere in the reply as a number (a checker that can be gamed)."""
    if kind == "strict":
        m = FINAL.search(reply.strip())
        return float(m is not None and int(m.group(1)) == answer)
    return float(str(answer) in re.findall(r"\d+", reply))

# ---------------------------------------------------------------- warm-start: SFT on the format


def direct(a, b):
    return f"{a} + {b} = {a + b}. The answer is {a + b}."


def worked(a, b):
    """Column addition, ones first, with the carries written out."""
    da, db, carry, steps = str(a)[::-1], str(b)[::-1], 0, []
    for i in range(max(len(da), len(db))):
        x, y = int(da[i]) if i < len(da) else 0, int(db[i]) if i < len(db) else 0
        s = x + y + carry
        terms = f"{x} + {y}" + (f" + {carry}" if carry else "")
        steps.append(f"{terms} = {s}, write {s % 10}" + (", carry 1" if s >= 10 else ""))
        carry = s // 10
    if carry:
        steps.append("Write the carry 1")
    return "Add the digits from right to left. " + ". ".join(steps) + f". The answer is {a + b}."


def warmstart(a):
    """Part 10's masked SFT on a.n synthetic problems, half answered directly, half worked, plus
    (with --chat N) the first N of Part 10's smol-smoltalk training conversations, so the model keeps
    chatting. Each step takes a.batch examples, cut into micro-batches of at most a.budget padded
    tokens, and divides by the step's count of trained tokens (Part 10's fix)."""
    from snapshots import load
    from sft import Conversations
    from train import lr_at, make_optimizer
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = chat_tokenizer()
    held = {(x, y) for s in test_set().values() for _, x, y in s}
    data = []
    for i, (q, x, y) in enumerate(problems(a.n, seed=a.seed, avoid=held)):
        reply = worked(x, y) if i % 2 else direct(x, y)
        data.append(render([{"role": "user", "content": q}, {"role": "assistant", "content": reply}], tok))
    if a.chat:
        conv = Conversations("train", a.chat)
        data += [tuple(v.tolist() for v in conv.example(i)) for i in range(conv.n)]
    a.steps = a.epochs * math.ceil(len(data) / a.batch)
    model = load(a.init).train()
    opt = make_optimizer(model, a)
    log, rng, step = open(os.path.join(a.out, "log.jsonl"), "a"), random.Random(a.seed), 0
    for _ in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for k in range(0, len(order), a.batch):
            group = sorted((data[i] for i in order[k:k + a.batch]), key=lambda e: len(e[0]))
            micro, j = [], 0
            while j < len(group):                             # pack by length into <= a.budget tokens
                n = 1
                while j + n < len(group) and (n + 1) * len(group[j + n][0]) <= a.budget:
                    n += 1
                micro.append(group[j:j + n]); j += n
            n_step = sum(sum(t != -1 for t in tg) for _, tg in group)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a)
            total = 0.0
            for mb in micro:
                T = max(len(ids) for ids, _ in mb)
                x = torch.zeros(len(mb), T, dtype=torch.long)
                y = torch.full((len(mb), T), -1, dtype=torch.long)
                for r, (ids, tg) in enumerate(mb):
                    x[r, :len(ids)], y[r, :len(tg)] = torch.tensor(ids), torch.tensor(tg)
                n = int((y != -1).sum())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x.cuda(), y.cuda()) * (n / n_step)   # this micro-batch's share
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.write(json.dumps({"step": step, "loss": total, "lr": lr_at(step, a), "examples": len(group),
                                  "micro": len(micro), "targets": n_step}) + "\n")
            step += 1
    torch.save(model.state_dict(), os.path.join(a.out, "model_final.pt"))

# ---------------------------------------------------------------- GRPO


def group_advantages(r, std=True, eps=1e-4):
    """r: (prompts, G) rewards. GRPO: (r - group mean) / group std. Dr. GRPO: r - group mean."""
    adv = r - r.mean(1, keepdim=True)
    return adv / (r.std(1, keepdim=True, unbiased=False) + eps) if std else adv


def token_logps(model, x, y, entropy=False):
    """Log-probability of every target token (0 where y is -1) for a right-padded batch, and, if
    asked, the entropy of each position's next-token distribution (no gradient)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    logp = -F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-1,
                            reduction="none").view_as(y)
    if not entropy:
        return logp
    with torch.no_grad():
        lp = F.log_softmax(logits.float(), -1)
        return logp, -(lp.exp() * lp).sum(-1)


def grpo_loss(logp, old, ref, adv, mask, eps=0.2, beta=0.04, agg="seq-mean", max_tokens=128):
    """Clipped policy-ratio loss on reply tokens plus beta * k3 KL to the reference.
    logp, old, ref, mask: (rows, T); adv: (rows,). agg="seq-mean" averages each reply over its own
    tokens, then over replies (GRPO as published); "token-sum" sums tokens and divides by a constant,
    max_tokens per reply (Dr. GRPO)."""
    ratio = torch.exp(logp - old)
    a = adv[:, None]
    pg = -torch.min(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    kl = torch.exp(ref - logp) - (ref - logp) - 1             # k3: never negative, averages to KL
    per_tok = (pg + beta * kl) * mask
    if agg == "seq-mean":
        loss = (per_tok.sum(1) / mask.sum(1).clamp(min=1)).mean()
    else:
        loss = per_tok.sum(1).mean() / max_tokens
    clipped = ((ratio - 1).abs() > eps) & mask.bool()
    return loss, kl.detach(), clipped


def collate_replies(prompts, replies):
    """Prompt + reply (+ <|end|> if it ended), right-padded; targets only on the reply tokens."""
    seqs = [(p + ids + ([END] if ended else []), len(p)) for p, (ids, ended) in zip(prompts, replies)]
    T = max(len(s) for s, _ in seqs) - 1
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    y = torch.full((len(seqs), T), -1, dtype=torch.long)
    for r, (s, st) in enumerate(seqs):
        x[r, :len(s) - 1] = torch.tensor(s[:-1])
        y[r, st - 1:len(s) - 1] = torch.tensor(s[st:])        # position t predicts token t + 1
    return x, y


def sample(model, prompts, max_new, temperature=1.0, seed=0, batch=256):
    """generate_batch over many prompts, `batch` rows at a time."""
    out = []
    for k in range(0, len(prompts), batch):
        out += generate_batch(model, prompts[k:k + batch], max_new, temperature, seed=seed + k)
    return out


def pass_at_k(n, c, k):
    """Chen et al.'s unbiased estimate of pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


@torch.no_grad()
def accuracy(model, tok, probs, kind="strict", max_new=128, batch=256):
    """Greedy accuracy on a list of (question, a, b), and the share of worked-style replies."""
    P = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
    replies = sample(model, P, max_new, temperature=0.0, batch=batch)
    texts = [tok.decode(ids) for ids, _ in replies]
    acc = sum(reward(t, x + y, kind) for t, (_, x, y) in zip(texts, probs)) / len(probs)
    return acc, sum(t.startswith("Add the digits") for t in texts) / len(texts)


def train(a):
    from snapshots import load
    from train import make_optimizer
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    tok, tests = chat_tokenizer(), test_set()
    held = {(x, y) for s in tests.values() for _, x, y in s}
    policy, ref = load(a.init).train(), load(a.init).eval()   # ref: the starting weights, frozen
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = make_optimizer(policy, a)
    step, ckpt_path = 0, os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        policy.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step = ck["step"]
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def write(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
    write({"event": "resume" if step else "start", "step": step, "args": vars(a), "torch": torch.__version__})
    t0 = time.time()
    while step < a.steps:
        if a.eval_every and step % a.eval_every == 0:
            policy.eval()
            acc, style = accuracy(policy, tok, tests["test"])          # always the strict checker
            rec = {"step": step, "test_acc": acc, "test_worked": style}
            if a.reward != "strict":
                rec["test_acc_" + a.reward] = accuracy(policy, tok, tests["test"], a.reward)[0]
            policy.train()
            write(rec)
        if a.time_cap and time.time() - t0 > a.time_cap:
            write({"event": "time_cap", "step": step}); break
        # 1. sample G replies to each of a.prompts new problems
        t = time.time()
        rng = random.Random(a.seed * 100_003 + step)             # same problems on resume
        probs = [problem(rng) for _ in range(a.prompts)]
        while any((x, y) in held for _, x, y in probs):
            probs = [p if (p[1], p[2]) not in held else problem(rng) for p in probs]
        P = [prompt_ids([{"role": "user", "content": q}], tok) for q, _, _ in probs]
        rows = [p for p in P for _ in range(a.group)]
        policy.eval()
        replies = sample(policy, rows, a.max_new, a.temperature, seed=a.seed * 1_000_003 + step * 4096,
                         batch=a.gen_batch)
        policy.train()
        torch.cuda.synchronize()
        t_gen = time.time() - t
        # 2. score, and group-relative advantages
        texts = [tok.decode(ids) for ids, _ in replies]
        answers = [x + y for _, x, y in probs for _ in range(a.group)]
        r = torch.tensor([reward(s, ans, a.reward) for s, ans in zip(texts, answers)]).view(a.prompts, a.group)
        adv = group_advantages(r, std=not a.no_std).flatten()
        x, y = collate_replies(rows, replies)
        mask = (y != -1).float()
        # 3. the old policy's and the reference's log-probs, once per batch of samples
        t = time.time()
        with torch.no_grad():
            old = torch.cat([token_logps(policy, x[k:k + a.micro].cuda(), y[k:k + a.micro].cuda())
                             for k in range(0, len(x), a.micro)])
            refp = torch.cat([token_logps(ref, x[k:k + a.micro].cuda(), y[k:k + a.micro].cuda())
                              for k in range(0, len(x), a.micro)])
        torch.cuda.synchronize()
        t_ref, t = time.time() - t, time.time()
        # 4. a.updates optimizer steps on disjoint minibatches of whole groups
        order = torch.randperm(a.prompts, generator=torch.Generator().manual_seed(step))
        stats = {"kl": 0.0, "entropy": 0.0, "clipped": 0.0, "tokens": 0.0, "loss": 0.0, "grad_norm": 0.0}
        for u in range(a.updates):
            groups = order[u::a.updates]
            idx = (groups[:, None] * a.group + torch.arange(a.group)).flatten()
            for k in range(0, len(idx), a.micro):
                i = idx[k:k + a.micro]
                xi, yi, mi = x[i].cuda(), y[i].cuda(), mask[i].cuda()
                logp, ent = token_logps(policy, xi, yi, entropy=True)
                loss, kl, clipped = grpo_loss(logp, old[i.cuda()], refp[i.cuda()], adv[i].cuda(), mi,
                                              a.clip, a.kl, a.agg, a.max_new)
                (loss * len(i) / len(idx)).backward()          # each minibatch: a mean over its replies
                stats["loss"] += loss.item() * len(i) / len(idx) / a.updates
                stats["kl"] += (kl * mi).sum().item()
                stats["entropy"] += (ent * mi).sum().item()
                stats["clipped"] += clipped.sum().item()
                stats["tokens"] += mi.sum().item()
            stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(policy.parameters(), a.grad_clip).item() / a.updates
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup) if a.warmup else a.lr
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_train = time.time() - t
        n = stats.pop("tokens")
        lens = [len(ids) for ids, _ in replies]
        rec = {"step": step, "reward": r.mean().item(),
               "zero_var": (r.std(1) == 0).float().mean().item(),
               "len": sum(lens) / len(lens), "ended": sum(e for _, e in replies) / len(replies),
               "worked": sum(s.startswith("Add the digits") for s in texts) / len(texts),
               "kl_tok": stats["kl"] / n, "entropy": stats["entropy"] / n, "clip_frac": stats["clipped"] / n,
               "loss": stats["loss"], "grad_norm": stats["grad_norm"], "t_gen": round(t_gen, 3),
               "t_ref": round(t_ref, 3), "t_train": round(t_train, 3),
               "samples_per_s": round(len(rows) / (t_gen + t_ref + t_train), 2),
               "elapsed": round(time.time() - t0, 1)}
        if step % a.show_every == 0:
            rec["examples"] = {"question": probs[0][0], "replies": texts[:a.group],
                               "rewards": r[0].tolist()}
        write(rec)
        step += 1
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(), "step": step},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    torch.save(policy.state_dict(), os.path.join(a.out, "model_final.pt"))   # weights only
    policy.eval()
    acc, style = accuracy(policy, tok, tests["test"])
    write({"step": step, "test_acc": acc, "test_worked": style})
    write({"event": "end", "step": step, "elapsed": round(time.time() - t0, 1),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)})


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["warmstart", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=os.path.join(HERE, "..", "t10", "runs", "sft88", "model_final.pt"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fused", action="store_true")
    # warmstart
    p.add_argument("--n", type=int, default=2000, help="warm-start problems")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64, help="warm-start examples per step")
    p.add_argument("--chat", type=int, default=0, help="also train on the first N smol-smoltalk conversations")
    p.add_argument("--budget", type=int, default=4096, help="warm-start padded tokens per micro-batch")
    # GRPO
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--prompts", type=int, default=32, help="problems per step")
    p.add_argument("--group", type=int, default=8, help="G: replies per problem")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--updates", type=int, default=1, help="optimizer steps per batch of samples")
    p.add_argument("--clip", type=float, default=0.2, help="PPO's epsilon")
    p.add_argument("--kl", type=float, default=0.04, help="beta, the KL coefficient")
    p.add_argument("--no-std", action="store_true", help="Dr. GRPO: do not divide by the group std")
    p.add_argument("--agg", choices=["seq-mean", "token-sum"], default="seq-mean")
    p.add_argument("--reward", choices=["strict", "contains"], default="strict")
    p.add_argument("--micro", type=int, default=32, help="rows per forward/backward pass")
    p.add_argument("--gen-batch", type=int, default=256, help="rows per sampling batch")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--show-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--time-cap", type=float, default=0, help="seconds; stop sampling new batches after")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    {"warmstart": warmstart, "train": train}[args.cmd](args)
