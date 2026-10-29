import torch, json
from circuit_tracer.graph import Graph
g = Graph.from_pt("runs/graph_dallas.pt")
A = g.adjacency_matrix
feats = g.active_features[g.selected_features].tolist()
acts = g.activation_values[g.selected_features].tolist()
idx = {tuple(f): i for i, f in enumerate(feats)}
n = A.shape[0]; nl = len(g.logit_targets); nf = len(feats); nt = len(g.input_tokens); L = 26
t = n - nl
def node(l, p, f): return idx.get((l, p, f))
out = {}
for name, (l, p, f) in {"texas20@10": (20, 10, 15589), "texas20@9": (20, 9, 15589), "say_capital21@10": (21, 10, 5943), "say_austin23@10": (23, 10, 12237), "texas14@9": (14, 9, 2268), "texas16@9": (16, 9, 25)}.items():
    i = node(l, p, f)
    out[name] = dict(i=i, act=acts[i] if i is not None else None, to_austin_logit=A[t, i].item() if i is not None else None)
for a, b in [("texas20@10", "say_austin23@10"), ("say_capital21@10", "say_austin23@10"), ("texas14@9", "texas20@10"), ("texas16@9", "texas20@10"), ("texas20@9", "texas20@10")]:
    out[f"{a}->{b}"] = A[out[b]["i"], out[a]["i"]].item()
# token (embedding) node for Dallas (pos 9): token nodes come after errors
tok_dallas = nf + nt * L + 9
out["emb_dallas->texas14@9"] = A[out["texas14@9"]["i"], tok_dallas].item()
out["emb_dallas->austin_logit"] = A[t, tok_dallas].item()
# share of incoming |edge| weight into the Austin logit from error nodes
inc = A[t].abs(); out["austin_in_from_errors"] = (inc[nf:nf + nt * L].sum() / inc.sum()).item()
out["austin_in_from_tokens"] = (inc[nf + nt * L:nf + nt * L + nt].sum() / inc.sum()).item()
print(json.dumps(out, indent=1))
json.dump(out, open("post/dallas_edges.json", "w"), indent=1)
