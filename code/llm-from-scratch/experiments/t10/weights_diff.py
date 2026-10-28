# How far SFT moved the weights: relative change per tensor group, and the special tokens' embedding rows.
import torch, json, sys
base = torch.load('/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt', map_location='cpu')
out = {}
for tag in sys.argv[1:]:
    sft = torch.load(f'/home/amit/.cache/vidyalai-scratch/t10/runs/{tag}/model_final.pt', map_location='cpu')
    d = {k: ((sft[k] - base[k]).norm() / base[k].norm()).item() for k in base}
    groups = {"embedding": [k for k in d if "tok_emb" in k], "attention": [k for k in d if ".attn." in k],
              "mlp": [k for k in d if ".mlp." in k], "norms": [k for k in d if "norm" in k]}
    tot = (sum((sft[k] - base[k]).norm() ** 2 for k in base) ** 0.5 / sum(base[k].norm() ** 2 for k in base) ** 0.5).item()
    E0, E1 = base['tok_emb.weight'], sft['tok_emb.weight']
    row = (E1 - E0).norm(dim=1)
    n = torch.nn.functional.normalize
    c = n(E1[1:5], dim=1) @ n(E1[1:5], dim=1).T
    nearest = {}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('/home/amit/.cache/vidyalai-scratch/fromscratch/tokenizer-32k.json')
    for i in range(0, 8):
        sims = n(E1, dim=1) @ n(E1[i], dim=0)
        sims[i] = -2
        top = sims.topk(5)
        nearest[tok.id_to_token(i)] = [(tok.decode([j]), round(v, 3)) for v, j in zip(top.values.tolist(), top.indices.tolist())]
    out[tag] = {"rel_change_total": tot,
                "rel_change_groups": {g: (sum((sft[k] - base[k]).norm() ** 2 for k in ks) ** 0.5 / sum(base[k].norm() ** 2 for k in ks) ** 0.5).item() for g, ks in groups.items()},
                "special_row_move": row[:8].tolist(), "median_row_move": row.median().item(),
                "p99_row_move": row.quantile(0.99).item(),
                "special_norms_after": E1[:8].norm(dim=1).tolist(),
                "chat_cos_after": c.tolist(), "nearest": nearest}
    print(tag, json.dumps(out[tag], indent=0)[:3000])
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t10/post/weights_diff.json', 'w'), indent=1)
