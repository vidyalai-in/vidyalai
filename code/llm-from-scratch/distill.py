"""Knowledge distillation and synthetic data (Build an LLM From Scratch, Part 13).

    python distill.py --model s15 --teacher ../t07/runs/main88/model_03360.pt --temperature 1 --alpha 1 \\
        --out runs/kd_s15 --steps 1000 --lr 2.08e-3 --warmup 45 --micro-batch 8 --accum 8 --compile --fused

Logit distillation trains a small student on the same token stream as Part 5's train.py, but each
target is the teacher's whole next-token distribution instead of one token id. That needs one
teacher forward pass per micro-batch and a shared vocabulary. Sequence-level distillation needs
neither: a teacher (any model, any tokenizer) writes replies, and the student is fine-tuned on
them with Part 10's sft.py. write_split() and sft_train() connect the two.
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- the loss


def kd_loss(student_logits, teacher_logits, targets, temperature=1.0, alpha=1.0):
    """Hinton et al.'s mix: alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(targets).
    Both distributions are softened by the same temperature T; the T^2 keeps the soft term's
    gradients the same size as T changes. Returns (loss, kl, ce), means over positions whose
    target is not -1. kl is reported at temperature T."""
    V = student_logits.size(-1)
    s = student_logits.float().view(-1, V)
    t = teacher_logits.float().view(-1, V)
    y = targets.reshape(-1)
    keep = y != -1
    log_q = F.log_softmax(s / temperature, dim=-1)                 # student, softened
    log_p = F.log_softmax(t / temperature, dim=-1)                 # teacher, softened
    kl_rows = (log_p.exp() * (log_p - log_q)).sum(-1)              # KL(p || q) per position
    kl = kl_rows[keep].mean()
    ce = F.cross_entropy(s, y, ignore_index=-1)                    # the hard labels, at T = 1
    loss = alpha * temperature ** 2 * kl + (1 - alpha) * ce
    return loss, kl, ce


def topk_mass(teacher_logits, ks=(1, 8, 32, 256)):
    """Mean share of the teacher's probability that its k most likely tokens hold, per k: what a
    precomputed top-k target file would keep."""
    p = F.softmax(teacher_logits.float().view(-1, teacher_logits.size(-1)), dim=-1)
    top = p.topk(max(ks), dim=-1).values.cumsum(-1)
    return {k: top[:, k - 1].mean().item() for k in ks}


def topk_storage_bytes(n_tokens, k, value_bytes=2, index_bytes=2):
    """Bytes to store k (probability, token id) pairs per training token. A 32,768-entry vocabulary
    needs ids 0..32767, which fit in 2 bytes unsigned; larger vocabularies need 3 or 4."""
    return n_tokens * k * (value_bytes + index_bytes)

# ---------------------------------------------------------------- logit distillation run


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default=os.path.join(HERE, "..", "t07", "runs", "main88", "model_03360.pt"))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0, help="weight of the soft (teacher) term; 0 = plain CE")
    kd, rest = p.parse_known_args(argv)
    from train import get_args as train_args                       # Part 5's flags, unchanged
    a = train_args(rest)
    for k, v in vars(kd).items():
        setattr(a, k, v)
    return a


@torch.no_grad()
def evaluate_kd(student, teacher, loader, n_batches, loss_fn=kd_loss):
    """Validation CE of the student and its KL to the teacher (T = 1), on the same tokens each time."""
    student.eval()
    ce = kl = 0.0
    for i in range(n_batches):
        x, y = (t.cuda(non_blocking=True) for t in loader.batch(i))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, t = student(x), teacher(x)
        _, k, c = loss_fn(s, t, y, 1.0, 1.0)
        ce, kl = ce + c.item(), kl + k.item()
    student.train()
    return ce / n_batches, kl / n_batches


def main(argv=None):
    from data import ShardLoader
    from model import GPT
    from snapshots import load
    from train import CONFIGS, lr_at, make_optimizer, save_checkpoint
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ckpt_path, log_path = os.path.join(a.out, "ckpt.pt"), os.path.join(a.out, "log.jsonl")
    torch.manual_seed(a.seed)
    raw = GPT(CONFIGS[a.model]).cuda()
    opt = make_optimizer(raw, a)
    step = 0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step = ck["step"]
        del ck
    teacher = load(a.teacher).eval().requires_grad_(False) if a.alpha > 0 else None
    student = torch.compile(raw) if a.compile else raw
    if teacher is not None and a.compile:
        teacher = torch.compile(teacher)
    loss_fn = torch.compile(kd_loss) if a.compile else kd_loss
    train = ShardLoader("train", B=a.micro_batch, T=a.seq_len)
    val = ShardLoader("val", B=a.micro_batch, T=a.seq_len)
    tps = a.micro_batch * a.seq_len * a.accum
    log = open(log_path, "a")
    def write(r):
        log.write(json.dumps(r) + "\n"); log.flush()
    write({"event": "resume" if step else "start", "step": step, "args": vars(a), "config": asdict(raw.cfg),
           "params": raw.num_params(), "tokens_per_step": tps, "torch": torch.__version__})
    batches = train.batches(step=step, accum=a.accum)
    stop, start, t0 = a.stop_at or a.steps, step, time.time()
    def validate():
        if teacher is None:                                       # plain CE: Part 5's evaluate
            from train import evaluate
            return {"val_loss": evaluate(student, val, a.val_batches, torch.autocast("cuda", dtype=torch.bfloat16))}
        ce, kl = evaluate_kd(student, teacher, val, a.val_batches, loss_fn)
        return {"val_loss": ce, "val_kl": kl}
    while step < stop:
        if a.val_every and step % a.val_every == 0 and (step == 0 or step != start):
            write({"step": step, **validate(), "tokens": step * tps})
        t_step, lr = time.time(), lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        sums = torch.zeros(3, device="cuda")
        t_teacher = 0.0
        for _ in range(a.accum):
            x, y = (t.pin_memory().cuda(non_blocking=True) for t in next(batches))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if teacher is None:
                    loss = student(x, y)
                    parts = torch.stack([loss.detach(), torch.zeros_like(loss), loss.detach()])
                else:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss, kl, ce = loss_fn(student(x), t_logits, y, a.temperature, a.alpha)
                    parts = torch.stack([loss.detach(), kl.detach(), ce.detach()])
            (loss / a.accum).backward()
            sums += parts / a.accum
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.grad_clip if a.grad_clip > 0 else float("inf"))
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_v, kl_v, ce_v = sums.tolist()                       # waits for the GPU
        dt = time.time() - t_step
        step += 1
        write({"step": step - 1, "loss": loss_v, "kl": kl_v, "ce": ce_v, "lr": lr, "grad_norm": gn.item(),
               "tokens": step * tps, "dt": round(dt, 4), "tok_per_s": round(tps / dt),
               "elapsed": round(time.time() - t0, 2)})
        if not math.isfinite(loss_v):
            write({"step": step - 1, "event": "diverged"}); break
        if step % a.ckpt_every == 0 or step == stop:
            save_checkpoint(ckpt_path, raw, opt, step, a)
    if step == a.steps:
        torch.save(raw.state_dict(), os.path.join(a.out, "model_final.pt"))   # weights only
    write({"step": step, **validate(), "tokens": step * tps})
    write({"event": "end", "step": step, "elapsed": round(time.time() - t0, 2),
           "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})

# ---------------------------------------------------------------- sequence-level distillation


def register_folded_gqa(name="sdpa_folded"):
    """An attention function for Hugging Face models with grouped-query attention, for decoding.
    With a padding mask, transformers' "sdpa" copies every key/value head n_rep times on every step
    (the masked GQA kernel would fall back to slow math). For one new token there's a cheaper exact
    trick: fold each group's n_rep query heads into the query-length axis, so they share one K/V head
    with no copy. Prompts (more than one query token) use the stock path. Load the model with
    attn_implementation=name."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

    def attn(module, query, key, value, attention_mask, scaling=None, **kw):
        B, Hq, T, D = query.shape
        Hkv = key.shape[1]
        if T != 1 or Hq == Hkv:
            return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)
        q = query.reshape(B, Hkv, Hq // Hkv, D)          # query head h reads K/V head h // n_rep
        out = F.scaled_dot_product_attention(q, key, value, attn_mask=attention_mask, scale=scaling)
        return out.reshape(B, Hq, 1, D).transpose(1, 2).contiguous(), None

    AttentionInterface.register(name, attn)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, sdpa_mask)
    return name


@torch.no_grad()
def teacher_generate(model, tok, conversations, max_new=512, batch_size=64, **sampling):
    """Replies from a Hugging Face chat model to a list of conversations (lists of {"role",
    "content"} ending with a user turn). Prompts are sorted by length and left-padded in batches.
    Returns one {"text", "finished", "new_tokens"} per conversation, in the input order."""
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    out = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **sampling)
        new = gen[:, enc["input_ids"].shape[1]:]
        for r, i in enumerate(idx):
            ids = new[r].tolist()
            ends = [k for k, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)]
            n = ends[0] if ends else len(ids)
            out[i] = {"text": tok.decode(ids[:n], skip_special_tokens=True), "finished": bool(ends),
                      "new_tokens": n}
    return out


def write_split(conversations, out_dir, split, sources=None, max_len=1024):
    """Tokenize conversations into sft.py's format (ids, is-target mask, offsets, sources), so that
    sft.Conversations can read them. Conversations that don't fit max_len, even cut to whole
    exchanges, are dropped. Returns the number kept."""
    from sft import chat_tokenizer, fit
    tok, ids, mask, offsets, kept = chat_tokenizer(), [], [], [0], []
    for n, msgs in enumerate(conversations):
        bodies = [tok.encode(m["content"].strip()).ids for m in msgs]
        x, y = fit(msgs, bodies, max_len)
        if x is None:
            continue
        ids += x; mask += [t != -1 for t in y]
        offsets.append(len(ids)); kept.append(sources[n] if sources else "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{split}_ids.npy", np.array(ids, dtype=np.uint16))
    np.save(f"{out_dir}/{split}_mask.npy", np.array(mask, dtype=bool))
    np.save(f"{out_dir}/{split}_offsets.npy", np.array(offsets, dtype=np.int64))
    json.dump(kept, open(f"{out_dir}/{split}_sources.json", "w"))
    return len(kept)


def sft_train(data_dir, argv):
    """Part 10's sft.py training run, reading its training split from data_dir instead of
    smol-smoltalk. The validation split is Part 10's, so every run is scored on the same set."""
    import sft
    for f in ("val_ids.npy", "val_mask.npy", "val_offsets.npy", "val_sources.json"):
        dst = os.path.join(data_dir, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(sft.DATA, f), dst)
    sft.DATA = data_dir
    sft.train(sft.get_args(["train"] + argv))


if __name__ == "__main__":
    main()
