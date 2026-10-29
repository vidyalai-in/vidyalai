# Check attribution graphs against interventions on the real model.
# 1. For the 50 most influential feature nodes of each graph: zero the feature at its position and
#    measure the change in the top target's (demeaned, pre-softcap) logit, with attention free
#    (the real model) and with attention patterns frozen. Compare with what the graph predicts.
# 2. Dallas -> Texas -> Austin: suppress "Texas" features, inject "California" features taken
#    from the Oakland prompt (feature IDs from circuit-tracer's Gemma demo), read the output.
import sys, json, time, resource, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab.attribution import load_gemma_tl, load_transcoders
from circuit_tracer.graph import Graph, compute_influence, normalize_matrix

torch.set_grad_enabled(False)
K = 50
t0 = time.time()
tcs = load_transcoders(lazy=True)
model = load_gemma_tl(tcs, n_ctx=64)
tok = model.tokenizer
out = {}


def demeaned_logit(logits, t):
    l = logits[0, -1].float()
    return (l[t] - l.mean()).item()


for name in ["dallas", "brandenburg", "ioi"]:
    g = Graph.from_pt(f"/home/amit/.cache/vidyalai-scratch/p08/runs/graph_{name}.pt")
    A = g.adjacency_matrix.float()
    n = A.shape[0]
    n_logits, n_feat = len(g.logit_targets), len(g.selected_features)
    t_node = n - n_logits                       # the top logit target
    t_tok = int(g.logit_token_ids[0])
    lw = torch.zeros(n); lw[-n_logits:] = g.logit_probabilities.cpu().float()
    infl = compute_influence(normalize_matrix(A), lw)
    feats = g.active_features[g.selected_features]              # [n_feat, 3] layer, pos, id
    acts = g.activation_values[g.selected_features].float()
    # linear prediction of zeroing node i: propagate per unit of activation through the graph
    J = A.clone()
    J[:, :n_feat] /= acts.clamp(min=1e-6)[None, :]
    r = torch.zeros(n); r[t_node] = 1.0; total = torch.zeros(n)
    for _ in range(64):
        total += r
        r = r @ J
        if r.abs().max() < 1e-7:
            break
    pred_total = -(acts * total[:n_feat])
    pred_direct = -A[t_node, :n_feat]
    order = infl[:n_feat].argsort(descending=True)[:K]
    prompt = g.input_string
    ids = g.input_tokens.to("cuda")[None]
    with model.zero_softcap():
        base = demeaned_logit(model(ids), t_tok)
        rows = []
        for i in order.tolist():
            layer, pos, fid = feats[i].tolist()
            row = dict(layer=layer, pos=pos, feature=fid, act=acts[i].item(), influence=infl[i].item(),
                       pred_total=pred_total[i].item(), pred_direct=pred_direct[i].item())
            for fz in (False, True):
                lg, _ = model.feature_intervention(ids[0], [(layer, pos, fid, 0.0)],
                                                   freeze_attention=fz, return_activations=False)
                row["actual_frozen" if fz else "actual"] = demeaned_logit(lg, t_tok) - base
            rows.append(row)
    out[name] = dict(target=tok.decode([t_tok]), base_demeaned_logit=base, nodes=rows)
    print(name, out[name]["target"], f"{time.time()-t0:.0f}s", flush=True)
    del g, A, J

# ---- two-hop swap, with the supernodes from circuit-tracer's demos/circuit_tracing_tutorial.ipynb
TEXAS = [(20, 9, 15589), (19, 9, 7477), (16, 9, 25), (4, 9, 13154), (14, 9, 2268), (7, 9, 6861)]
CALIFORNIA = [(22, 10, 4367), (21, 10, 2464), (6, 9, 13909), (8, 9, 14641), (14, 9, 12562)]
dallas = "Fact: the capital of the state containing Dallas is"
oakland = "Fact: the capital of the state containing Oakland is"


def top5(logits):
    p = logits[0, -1].float().softmax(-1)
    v, i = p.topk(5)
    return [(tok.decode([j]), round(x, 4)) for x, j in zip(v.tolist(), i.tolist())]


def p_of(logits, word):
    return logits[0, -1].float().softmax(-1)[tok.encode(word, add_special_tokens=False)[0]].item()


lg_d, act_d = model.get_activations(dallas)
lg_o, act_o = model.get_activations(oakland)
swap = {"dallas": dict(top5=top5(lg_d), p_austin=p_of(lg_d, " Austin"), p_sacramento=p_of(lg_d, " Sacramento")),
        "oakland": dict(top5=top5(lg_o)),
        "texas_acts": [act_d[l, p, f].item() for l, p, f in TEXAS],
        "california_acts_on_oakland": [act_o[l, p, f].item() for l, p, f in CALIFORNIA],
        "california_acts_on_dallas": [act_d[l, p, f].item() for l, p, f in CALIFORNIA]}
for fz in (False, True):
    for label, iv in [
        ("texas_x-2", [(l, p, f, -2 * act_d[l, p, f].item()) for l, p, f in TEXAS]),
        ("texas_0", [(l, p, f, 0.0) for l, p, f in TEXAS]),
        ("texas_x-2_california_x2", [(l, p, f, -2 * act_d[l, p, f].item()) for l, p, f in TEXAS] +
                                    [(l, p, f, 2 * act_o[l, p, f].item()) for l, p, f in CALIFORNIA]),
        ("california_x2_only", [(l, p, f, 2 * act_o[l, p, f].item()) for l, p, f in CALIFORNIA]),
    ]:
        lg, _ = model.feature_intervention(dallas, iv, freeze_attention=fz, return_activations=False)
        swap[f"{label}{'_frozen' if fz else ''}"] = dict(top5=top5(lg), p_austin=p_of(lg, " Austin"),
                                                        p_sacramento=p_of(lg, " Sacramento"))
# random control: 6 random active features at the same positions, x-2
g = torch.Generator().manual_seed(0)
ctrl = []
for s in range(20):
    iv = []
    for l, p, f in TEXAS:
        active = act_d[l, p].nonzero()[:, 0]
        ff = active[torch.randint(len(active), (1,), generator=g)].item()
        iv.append((l, p, ff, -2 * act_d[l, p, ff].item()))
    lg, _ = model.feature_intervention(dallas, iv, freeze_attention=False, return_activations=False)
    ctrl.append(p_of(lg, " Austin"))
swap["random6_x-2_p_austin"] = ctrl
out["swap"] = swap
out["seconds"] = time.time() - t0
out["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
out["host_peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
json.dump(out, open("/home/amit/.cache/vidyalai-scratch/p08/post/validate.json", "w"), indent=1)
print(json.dumps(swap, indent=1))
print(out["seconds"], out["gpu_peak_gib"], out["host_peak_gib"])
