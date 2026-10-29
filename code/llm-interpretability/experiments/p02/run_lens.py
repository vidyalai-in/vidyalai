"""Tuned-lens experiments for Part 2. Usage: run_lens.py <model> <phase> [args]
phases: sweep_bias, sweep_muon <bias_lr>, train <kind> <opt> <steps> <lr> <bias_lr> <tag>, eval <tag,...>
"""
import sys, json, time, torch, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import lens as LL

torch.backends.cuda.matmul.allow_tf32 = True      # fp32 weights; TF32 tensor-core matmuls
torch.backends.cudnn.allow_tf32 = True
R = "/home/amit/.cache/vidyalai-scratch/p02/runs"
HF = {"gpt2": "gpt2", "pythia-160m": "EleutherAI/pythia-160m", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B"}
key, phase = sys.argv[1], sys.argv[2]
d = torch.load(f"{R}/tokens-{key}.pt")
dev_chunks, test_chunks = d["eval"][:50], d["eval"][50:]     # 50 dev docs, 304 test docs
m = LL.load(HF[key], dtype=torch.bfloat16 if key == "qwen2.5-0.5b" else torch.float32)  # Qwen in its native bf16
B = 8


def batches(seed=0):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(d["train"]), generator=g)
    for i in range(0, len(order) - B + 1, B):
        yield d["train"][order[i:i + B]]


@torch.no_grad()
def dev_kl(lens, layers, chunks=dev_chunks):
    tot = torch.zeros(len(layers))
    for i in range(0, len(chunks), B):
        tok = chunks[i:i + B].cuda()
        logits, H = LL.resid_points(m, tok)
        flp = F.log_softmax(logits[:, 1:-1].float(), -1)
        for j, l in enumerate(layers):
            z = LL.logit_lens(m, lens.translate(l, H[l][:, 1:-1]) if lens else H[l][:, 1:-1])
            tot[j] += LL.kl_from_final(flp, z).mean().item() * len(tok)
    return (tot / len(chunks)).tolist()


t0 = time.time()
if phase == "sweeps":
    layers = [3, 7] if m.cfg.n_layers == 12 else [6, 14]
    res = {"layers": layers, "steps": 60, "logit": dev_kl(None, layers), "bias": [], "muon": []}
    for lr in [1e-3, 1e-2, 5e-2]:
        lens, _ = LL.train_tuned_lens(m, batches(), kind="bias", opt="adam", steps=60, layers=layers, bias_lr=lr)
        res["bias"].append({"lr": lr, "dev_kl": dev_kl(lens, layers)}); print(json.dumps(res["bias"][-1]), flush=True)
    blr = min(res["bias"], key=lambda r: sum(r["dev_kl"]))["lr"]
    res["best_bias_lr"] = blr
    for lr in [0.005, 0.02, 0.08]:
        lens, _ = LL.train_tuned_lens(m, batches(), kind="full", opt="muon", steps=60, layers=layers, lr=lr, bias_lr=blr)
        res["muon"].append({"lr": lr, "dev_kl": dev_kl(lens, layers)}); print(json.dumps(res["muon"][-1]), flush=True)
    for lr in [1.0]:
        lens, _ = LL.train_tuned_lens(m, batches(), kind="full", opt="sgd", steps=60, layers=layers, lr=lr)
        res["sgd"] = [{"lr": lr, "dev_kl": dev_kl(lens, layers)}]; print(json.dumps(res["sgd"][-1]), flush=True)
    res["sec"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{R}/{key}-sweeps.json", "w"), indent=1)
elif phase == "train":
    kind, opt, steps, lr, blr, tag = sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
    L = m.cfg.n_layers
    ev = lambda lens: {"dev_kl": dev_kl(lens, list(range(L)), dev_chunks[:24])}
    torch.cuda.reset_peak_memory_stats()
    lens, log = LL.train_tuned_lens(m, batches(), kind=kind, opt=opt, steps=steps, lr=lr, bias_lr=blr,
                                    eval_at={10, 25, 50, 100, 150, 200}, eval_fn=ev)
    torch.save(lens.state_dict(), f"{R}/{key}-lens-{tag}.pt")
    meta = {"model": HF[key], "kind": kind, "opt": opt, "steps": steps, "lr": lr, "bias_lr": blr,
            "batch_tokens": B * 512, "train_tokens": steps * B * 512, "sec": round(time.time() - t0, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "log": log}
    json.dump(meta, open(f"{R}/{key}-train-{tag}.json", "w"), indent=1)
elif phase == "eval":
    lenses = {}
    for tag in (sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []):
        sd = torch.load(f"{R}/{key}-lens-{tag}.pt")
        kind = "full" if any(k.startswith("weight") for k in sd) else "bias"
        lens = LL.TunedLens(m.cfg.n_layers, m.cfg.d_model, kind).cuda(); lens.load_state_dict(sd)
        lenses[tag] = lens
    bs = 2 if m.cfg.d_vocab > 100000 else 8
    out = LL.evaluate(m, test_chunks, lenses, batch=bs)
    out["sec"] = round(time.time() - t0, 1)
    out["n_docs"] = len(test_chunks); out["positions_per_doc"] = test_chunks.shape[1] - 2
    json.dump(out, open(f"{R}/{key}-eval.json", "w"))
if phase == "examples":
    prompts = json.load(open("/home/amit/.cache/vidyalai-scratch/p02/scripts/prompts.json"))
    lenses = {"logit": None}
    for tag in sys.argv[3].split(","):
        sd = torch.load(f"{R}/{key}-lens-{tag}.pt")
        kind = "full" if any(k.startswith("weight") for k in sd) else "bias"
        lens = LL.TunedLens(m.cfg.n_layers, m.cfg.d_model, kind).cuda(); lens.load_state_dict(sd); lenses[tag] = lens
    res = []
    with torch.no_grad():
        for pr, target in prompts:
            tok = m.to_tokens(pr, prepend_bos=True)
            tid = m.to_single_token(target)
            logits, H = LL.resid_points(m, tok)
            row = {"prompt": pr, "target": target, "n_tokens": tok.shape[1]}
            for n, lens in lenses.items():
                pts = []
                for l in range(H.shape[0]):
                    h = H[l][:, -1]
                    z = LL.logit_lens(m, lens.translate(l, h) if lens is not None else h)[0].float()
                    pz = z.softmax(-1); top = int(pz.argmax())
                    pts.append({"top": m.to_string(top), "p_top": round(float(pz[top]), 4),
                                "rank_target": int((z > z[tid]).sum()) + 1, "p_target": round(float(pz[tid]), 4)})
                row[n] = pts
            res.append(row)
    json.dump(res, open(f"{R}/{key}-examples.json", "w"), indent=1)
print("phase", phase, "done in", round(time.time() - t0, 1), "s")
