"""Part 6 toy-model experiments. Usage: toy_runs.py <exp> [steps]
exp: sweep | phase2 | phase3 | geometry | pentagon | neurons
Writes runs/<exp>.json (small) and runs/<exp>_W.pt (weights, not committed)."""
import json, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
import torch
from lab import toy

exp = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
dev = "cuda"
torch.manual_seed(0)


def logspace(a, b, n):
    return torch.logspace(torch.log10(torch.tensor(a)), torch.log10(torch.tensor(b)), n).tolist()


def finish(model, meta, extra):
    t = toy.train(model, steps=steps, log_every=steps // 4)
    loss = toy.eval_loss(model)
    W = model.W.detach()
    torch.save(dict(W=W.cpu(), b=model.b.detach().cpu(), loss=loss), RUN + exp + "_W.pt")
    out = dict(exp=exp, steps=steps, batch=1024, lr=1e-3, optimizer="AdamW wd=0.01", train_seconds=round(t, 1),
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, **meta)
    out.update(extra(model, W, loss))
    json.dump(out, open(RUN + exp + ".json", "w"))
    torch.save(dict(W=W.cpu(), b=model.b.detach().cpu(), loss=loss), RUN + exp + "_W.pt")
    print(exp, "done", round(t, 1), "s; peak GPU MB", torch.cuda.max_memory_allocated() // 2**20)


if exp in ("sweep", "sweep_linear"):
    # Elhage et al.'s first experiment: n=20 features, m=5 dims, importance 0.7^i, sweep sparsity
    n, m, reps = 20, 5, 5
    dens = logspace(1.0, 0.001, 13)
    D = torch.tensor(dens).repeat_interleave(reps)[:, None].expand(-1, n)
    imp = (0.7 ** torch.arange(n).float())[None].expand(len(D), n)
    kind = "linear" if exp == "sweep_linear" else "relu_out"
    model = toy.ToyModel(len(D), n, m, kind=kind, importance=imp, density=D, seed=1, device=dev)

    def extra(model, W, loss):
        c = toy.classify(W)
        return dict(densities=dens, reps=reps, n=n, m=m, importance="0.7^i",
                    learned=(c > 0).sum(1).tolist(), superposed=(c == 1).sum(1).tolist(),
                    classes=c.tolist(), norms=toy.norms(W).tolist(), dim=toy.dimensionality(W).tolist(),
                    loss=loss.tolist())
    finish(model, dict(kind=kind), extra)

elif exp in ("phase2", "phase3"):
    # one "extra" feature (the last) of varying importance; all features share one density
    n, m = (2, 1) if exp == "phase2" else (3, 2)
    reps = 10
    imps, dens = logspace(0.1, 10.0, 21), logspace(1.0, 0.01, 21)
    grid = [(i, d) for d in dens for i in imps]
    N = len(grid) * reps
    imp = torch.ones(N, n); den = torch.ones(N, n)
    for k, (i, d) in enumerate(grid):
        imp[k * reps:(k + 1) * reps, -1] = i
        den[k * reps:(k + 1) * reps] = d
    model = toy.ToyModel(N, n, m, kind="relu_out", importance=imp, density=den, seed=2, device=dev)

    def extra(model, W, loss):
        c = toy.classify(W)[:, -1].cpu().view(len(grid), reps)
        L = loss.view(len(grid), reps)
        keep = L.argsort(1)[:, :-1]                       # drop the highest-loss replicate, as the paper does
        ck = torch.gather(c, 1, keep)
        frac = torch.stack([(ck == j).float().mean(1) for j in range(3)], 1)  # not learned / superposition / dedicated
        return dict(n=n, m=m, reps=reps, importances=imps, densities=dens,
                    frac=frac.tolist(), majority=frac.argmax(1).tolist(), loss=L.tolist())
    finish(model, dict(kind="relu_out"), extra)

elif exp == "geometry":
    # uniform superposition: n=400, m=30, equal importance, sweep sparsity (paper's setup)
    n, m, reps = 400, 30, 3
    dens = logspace(1.0, 0.001, 30)
    D = torch.tensor(dens).repeat_interleave(reps)[:, None].expand(-1, n)
    model = toy.ToyModel(len(D), n, m, kind="relu_out", density=D, seed=3, device=dev)

    def extra(model, W, loss):
        return dict(n=n, m=m, reps=reps, densities=dens, dstar=toy.dims_per_feature(W).tolist(),
                    dim=[[round(v, 4) for v in r] for r in toy.dimensionality(W).tolist()],
                    norms=[[round(v, 4) for v in r] for r in toy.norms(W).tolist()], loss=loss.tolist())
    finish(model, dict(kind="relu_out"), extra)

elif exp == "pentagon":
    n, m, reps = 5, 2, 50
    dens = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    D = torch.tensor(dens).repeat_interleave(reps)[:, None].expand(-1, n)
    model = toy.ToyModel(len(D), n, m, kind="relu_out", density=D, seed=4, device=dev)

    def extra(model, W, loss):
        return dict(n=n, m=m, reps=reps, densities=dens, W=W.tolist(), b=model.b.detach().tolist(),
                    learned=(toy.norms(W) >= 0.5).sum(1).tolist(), loss=loss.tolist(),
                    dim=toy.dimensionality(W).tolist())
    finish(model, dict(kind="relu_out"), extra)

elif exp == "neurons":
    # ReLU hidden layer model, n=10 features, m=5 neurons, importance 0.75^i (paper's privileged-basis setup)
    n, m, reps = 10, 5, 500
    dens = logspace(1.0, 0.001, 8)
    D = torch.tensor(dens).repeat_interleave(reps)[:, None].expand(-1, n)
    imp = (0.75 ** torch.arange(n).float())[None].expand(len(D), n)
    model = toy.ToyModel(len(D), n, m, kind="relu_hidden", importance=imp, density=D, seed=5, device=dev)

    def extra(model, W, loss):
        nf = {str(t): toy.neuron_features(W, t).tolist() for t in (0.2, 0.3, 0.4)}
        best = loss.view(len(dens), reps).argmin(1) + torch.arange(len(dens)) * reps
        return dict(n=n, m=m, reps=reps, densities=dens, importance="0.75^i", neuron_features=nf,
                    learned=(toy.norms(W) >= 0.5).sum(1).tolist(), loss=loss.tolist(),
                    best=best.tolist(), W_best=W[best].tolist())
    finish(model, dict(kind="relu_hidden"), extra)
