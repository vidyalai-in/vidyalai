# Experiment B: model diffing with a base-model SAE. Runs the Gemma Scope layer-12 residual SAE
# (google/gemma-scope-2b-pt-res, width 16k, average L0 82; trained on gemma-2-2b) on the residual stream
# after decoder layer 12 of either gemma-2-2b (base) or gemma-2-2b-it, over the same token sequences:
#   plain: 300 held-out FineWeb-Edu documents x 256 tokens (Part 7's eval set)
#   chat:  256 chat-formatted conversations; the user turn is a FineWeb-Edu passage or an ARC-Easy question,
#          the model turn is Gemma 2 2B IT's own greedy reply (generated once, in the "gen" step).
# Usage: python diff_collect.py gen | base | it
import json, sys, time, numpy as np, torch
from lab import safety as S
from lab.sae import load_gemma_scope, TopK

D = "/home/amit/.cache/vidyalai-scratch/p11/data/diff"
LAYER = 12          # decoder layer index; its output is hidden_states[13]


def make_chat_prompts():
    from datasets import load_dataset
    T = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.IT)
    ev = T["eval"][300:428]          # docs not used in the plain set
    users = []
    for row in ev:
        passage = tok.decode(row[1:97].tolist()).strip()
        users.append("Explain the main point of this passage in simple words:\n\n" + passage)
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for r in arc.select(range(128)):
        opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        users.append(r["question"] + "\n" + opts + "\nExplain your reasoning briefly, then give the answer.")
    return users


def gen(model, tok, batch=32, max_new=128):
    """Gemma 2 2B IT's own greedy replies, generated in left-padded batches."""
    users = make_chat_prompts()
    tok.padding_side = "left"
    stop = {tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id}
    seqs, nprompt = [], []
    for s0 in range(0, len(users), batch):
        ps = [S.chat(u) for u in users[s0:s0 + batch]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for j in range(len(ps)):
            prompt = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            new = out[j, enc.input_ids.shape[1]:].cpu().tolist()
            cut = next((k + 1 for k, t in enumerate(new) if t in stop), len(new))
            seqs.append(torch.cat([prompt, torch.tensor(new[:cut])])); nprompt.append(len(prompt))
    tok.padding_side = "right"
    torch.save({"seqs": seqs, "nprompt": nprompt, "users": users}, f"{D}/chat_tokens.pt")
    print("generated", len(seqs), "mean len", np.mean([len(s) for s in seqs]), "mean reply", np.mean([len(s) - n for s, n in zip(seqs, nprompt)]), flush=True)


def categories(tok, ids, nprompt):
    """0 = BOS (skipped), 1 = template token (<start_of_turn>, <end_of_turn>, role names and their newline),
    2 = user text, 3 = model text. For plain documents everything after BOS is 4."""
    st, et = tok.convert_tokens_to_ids("<start_of_turn>"), tok.convert_tokens_to_ids("<end_of_turn>")
    c = np.array([0] + [2 if i < nprompt else 3 for i in range(1, len(ids))])
    for i in range(1, len(ids)):
        if ids[i] == st:
            c[i:i + 3] = 1                       # <start_of_turn>, the role name, the newline
        elif ids[i] == et:
            c[i:i + 2] = 1                       # <end_of_turn> and the newline after it
    return c


@torch.no_grad()
def run(which, model=None, tok=None):
    if model is None:
        model, tok = S.load_gemma(S.BASE if which == "base" else S.IT)
    sae = load_gemma_scope(layer=LAYER, width="16k", l0=82)
    state = {"mode": "grab", "x": None, "xhat": None}

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if state["mode"] == "grab":
            state["x"] = hs.detach()
            return None
        new = hs.clone()
        new[:, 1:] = state["xhat"][:, 1:].to(hs.dtype)   # splice the reconstruction everywhere but BOS
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
    model.model.layers[LAYER].register_forward_hook(hook)

    T = torch.load("/home/amit/.cache/vidyalai-scratch/interp/data/p07/gemma-tokens.pt")
    C = torch.load(f"{D}/chat_tokens.pt")
    sets = {"plain": [(row.long(), 1) for row in T["eval"][:300]],
            "chat": [(s.long(), n) for s, n in zip(C["seqs"], C["nprompt"])]}
    out = {}
    for name, seqs in sets.items():
        n = sae.W_enc.shape[1]
        cats = [1, 2, 3] if name == "chat" else [4]
        lat_sum = {c: torch.zeros(n, device="cuda", dtype=torch.float64) for c in cats}
        lat_cnt = {c: torch.zeros(n, device="cuda", dtype=torch.float64) for c in cats}
        top = TopK(n, 8)
        per_tok = []          # rows: seq, pos, cat, |x|^2, err^2, l0, ce_clean, ce_spliced
        sx = torch.zeros(sae.W_enc.shape[0], device="cuda", dtype=torch.float64)
        resid = []
        offset = 0
        t1 = time.time()
        for si, (ids, npr) in enumerate(seqs):
            x_ids = ids[None].cuda()
            state["mode"] = "grab"
            lg = model(input_ids=x_ids, use_cache=False).logits.float()
            x = state["x"][0].float()
            f, xh = S.sae_encode(sae, x)
            state["xhat"] = xh[None]; state["mode"] = "splice"
            lg2 = model(input_ids=x_ids, use_cache=False).logits.float()
            tgt = x_ids[0, 1:]
            ce1 = torch.nn.functional.cross_entropy(lg[0, :-1], tgt, reduction="none")
            ce2 = torch.nn.functional.cross_entropy(lg2[0, :-1], tgt, reduction="none")
            cat = categories(tok, ids.tolist(), npr) if name == "chat" else np.r_[0, np.full(len(ids) - 1, 4)]
            ct = torch.tensor(cat, device="cuda")
            for c in cats:
                m = ct == c
                lat_sum[c] += f[m].sum(0).double(); lat_cnt[c] += (f[m] > 0).sum(0).double()
            keep = ct > 0
            fk = f.clone(); fk[~keep] = 0
            top.update(fk, offset)
            sx += x[keep].sum(0).double()
            ce1p = torch.cat([ce1, torch.tensor([float("nan")], device="cuda")])   # loss predicting token i+1 from i
            ce2p = torch.cat([ce2, torch.tensor([float("nan")], device="cuda")])
            rows = torch.stack([torch.full_like(ce1p, si), torch.arange(len(ids), device="cuda").float(), ct.float(),
                                (x ** 2).sum(-1), ((x - xh) ** 2).sum(-1), (f > 0).sum(-1).float(), ce1p, ce2p], 1)
            per_tok.append(rows.cpu().numpy())
            if name == "chat":
                resid.append(x.half().cpu().numpy())
            offset += len(ids)
        P = np.concatenate(per_tok)
        np.save(f"{D}/{which}_{name}_tok.npy", P)
        torch.save({"sum": {c: v.cpu() for c, v in lat_sum.items()}, "cnt": {c: v.cpu() for c, v in lat_cnt.items()},
                    "top_v": top.v.cpu(), "top_i": top.i.cpu(), "count": top.count.cpu(), "sx": sx.cpu(),
                    "lens": [len(s[0]) for s in seqs]}, f"{D}/{which}_{name}_lat.pt")
        if resid:
            np.save(f"{D}/{which}_{name}_resid.npy", np.concatenate(resid))
        k = P[:, 2] > 0
        mean = (sx / k.sum()).cpu().numpy()
        var_tot = P[k, 3].sum() - k.sum() * (mean ** 2).sum()
        fve = 1 - P[k, 4].sum() / var_tot
        ok = k & ~np.isnan(P[:, 6])
        print(which, name, "seqs", len(seqs), "tokens", int(k.sum()), "s", round(time.time() - t1, 1),
              "FVE", round(float(fve), 4), "L0", round(float(P[k, 5].mean()), 1),
              "CE", round(float(P[ok, 6].mean()), 4), "CE spliced", round(float(P[ok, 7].mean()), 4), flush=True)
    print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    import os
    os.makedirs(D, exist_ok=True)
    if sys.argv[1] == "gen_it":          # generate the chat set, then run the IT model on everything
        t0 = time.time()
        model, tok = S.load_gemma(S.IT)
        gen(model, tok)
        print("gen s", round(time.time() - t0, 1), flush=True)
        run("it", model, tok)
    else:
        run(sys.argv[1])
