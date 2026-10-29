# How well do the 26 Gemma Scope transcoders stand in for Gemma 2 2B's MLPs?
# Per layer: fraction of variance unexplained (FVU) of the MLP output and L0, with the transcoder
# fed the pre-MLP RMSNorm output without its learned gain (circuit-tracer's ln2.hook_normalized)
# and with it. Whole model: next-token loss, top-1 agreement and KL when every MLP is replaced.
import sys, json, time, resource, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.attribution import load_gemma_tl, load_transcoders

torch.set_grad_enabled(False)
N_DOCS, LEN, BS = int(sys.argv[1]) if len(sys.argv) > 1 else 64, 128, 8
t0 = time.time()
model = load_gemma_tl(None, n_ctx=LEN + 1)
tcs = load_transcoders(lazy=True)
L = model.cfg.n_layers
tok = model.tokenizer
docs = []
for line in open("/home/amit/.cache/vidyalai-scratch/interp/data/p02/fineweb-edu-heldout600.jsonl"):
    ids = tok(json.loads(line)["text"]).input_ids          # starts with <bos>
    if len(ids) >= LEN + 1:
        docs.append(ids[:LEN + 1])
    if len(docs) == N_DOCS:
        break
X = torch.tensor(docs, device="cuda")
print(f"{len(docs)} docs x {LEN} predicted tokens; load {time.time()-t0:.0f}s", flush=True)

stats = {k: torch.zeros(L, dtype=torch.float64) for k in
         ["sse_nogain", "sse_gain", "sst", "l0_nogain", "l0_gain", "n"]}
ce_orig, ce_rep, agree, kl, agree_hard, n_hard = [], [], [], [], [], 0
for b in range(0, len(docs), BS):
    x = X[b:b + BS]
    names = lambda n: n.endswith("ln2.hook_normalized") or n.endswith("hook_mlp_out")
    logits, cache = model.run_with_cache(x, names_filter=names)
    for l in range(L):
        xin = cache[f"blocks.{l}.ln2.hook_normalized"][:, 1:]          # skip <bos>
        y = cache[f"blocks.{l}.hook_mlp_out"][:, 1:].float().flatten(0, 1)
        tc = tcs[l]
        for tag, inp in [("nogain", xin), ("gain", xin * model.blocks[l].ln2.w)]:
            a = tc.encode(inp.to(torch.bfloat16))
            yhat = tc.decode(a).float().flatten(0, 1)
            stats[f"sse_{tag}"][l] += ((y - yhat) ** 2).sum().item()
            stats[f"l0_{tag}"][l] += (a > 0).sum().item()
        stats["sst"][l] += ((y - y.mean(0)) ** 2).sum().item()   # per-batch mean (8 x 128 tokens)
        stats["n"][l] += y.shape[0]
    del cache

    held = {}
    def keep(act, hook):
        held[hook.layer()] = act
    def swap(act, hook):
        l = hook.layer()
        out = tcs[l].decode(tcs[l].encode(held.pop(l).to(torch.bfloat16)))
        out[:, 0] = act[:, 0]                                     # leave <bos> untouched
        return out.to(act.dtype)
    hooks = [(f"blocks.{l}.ln2.hook_normalized", keep) for l in range(L)] + \
            [(f"blocks.{l}.hook_mlp_out", swap) for l in range(L)]
    rlogits = model.run_with_hooks(x, fwd_hooks=hooks)
    for i in range(x.shape[0]):                       # one sequence at a time: [128, 256000] fp32
        lp, rlp = logits[i, :-1].float().log_softmax(-1), rlogits[i, :-1].float().log_softmax(-1)
        tgt = x[i, 1:]
        ce_orig.append(-lp.gather(-1, tgt[:, None])[:, 0].cpu())
        ce_rep.append(-rlp.gather(-1, tgt[:, None])[:, 0].cpu())
        same = (lp.argmax(-1) == rlp.argmax(-1))
        agree.append(same.cpu())
        kl.append((lp.exp() * (lp - rlp)).sum(-1).cpu())
        # Anthropic's filter: tokens the model predicts correctly with confidence below 80%
        p_top = lp.exp().max(-1).values
        hard = (lp.argmax(-1) == tgt) & (p_top < 0.8)
        agree_hard.append(same[hard].cpu())
        del lp, rlp
    del logits, rlogits
    print(f"batch {b // BS} {time.time()-t0:.0f}s", flush=True)

ce_orig, ce_rep, agree, kl = map(torch.cat, (ce_orig, ce_rep, agree, kl))
agree_hard = torch.cat(agree_hard)
docs_n = len(docs)

def boot(v, per_doc=LEN, n=2000):
    # bootstrap over documents
    v = v.view(docs_n, per_doc).double()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, docs_n, (n, docs_n), generator=g)
    m = v.mean(1)[idx].mean(1)
    return [v.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()]

res = dict(
    n_docs=docs_n, tokens=docs_n * LEN,
    per_layer=[dict(layer=l,
                    fvu_nogain=(stats["sse_nogain"][l] / stats["sst"][l]).item(),
                    fvu_gain=(stats["sse_gain"][l] / stats["sst"][l]).item(),
                    l0_nogain=(stats["l0_nogain"][l] / stats["n"][l]).item(),
                    l0_gain=(stats["l0_gain"][l] / stats["n"][l]).item()) for l in range(L)],
    ce_orig=boot(ce_orig), ce_rep=boot(ce_rep), ce_delta=boot(ce_rep - ce_orig),
    top1_agree=boot(agree.float()), kl_nats=boot(kl),
    top1_agree_hard=[agree_hard.float().mean().item(), int(agree_hard.numel())],
    seconds=time.time() - t0,
    gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
json.dump(res, open("/home/amit/.cache/vidyalai-scratch/p08/post/fidelity.json", "w"), indent=1)
for r in res["per_layer"]:
    print(r)
print({k: v for k, v in res.items() if k != "per_layer"})
