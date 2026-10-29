"""Which tokens fire each GPT-2 small MLP neuron? Same windows and thresholds as s6 (act > 0.5 * neuron max).
Token strings are normalized (strip spaces, lowercase) so ' Per', 'per' and ' per' count as one type.
Saves, per (neuron, domain, token type), the number of firing tokens: runs/s7_events.pt."""
import sys, time
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
RUN = "/home/amit/.cache/vidyalai-scratch/p06/runs/"
t0 = time.time()
model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cuda")
tok = model.tokenizer
win = torch.load(RUN + "s6_windows.pt")
z = torch.load(RUN + "s6_counts.pt")
W = win["tokens"].cuda()
tau = z["tau"].cuda()                                       # [L, H]
NW, T, nd = z["NW"], z["T"], len(z["doms"])
vocab = tok.vocab_size
norm = {}
norm_id = torch.empty(vocab, dtype=torch.long)
names_list = []
for i in range(vocab):
    s = tok.decode([i]).strip().lower()
    if s not in norm:
        norm[s] = len(norm); names_list.append(s)
    norm_id[i] = norm[s]
norm_id = norm_id.cuda()
V = len(norm)
L, H, B = 12, 3072, 20
names = [f"blocks.{l}.mlp.hook_post" for l in range(L)]
keys = []
for i in range(0, len(W), B):
    _, cache = model.run_with_cache(W[i:i + B], names_filter=lambda n: n in names)
    a = torch.stack([cache[n][:, 1:] for n in names])       # [L, b, T, H]
    l, b, t, h = (a > tau[:, None, None, :]).nonzero(as_tuple=True)
    d = (i + b) // NW
    tid = norm_id[W[i + b, t + 1]]
    keys.append(((l * H + h) * nd + d) * V + tid)
keys = torch.cat(keys)
uk, cnt = keys.unique(return_counts=True)
torch.save(dict(keys=uk.cpu(), counts=cnt.int().cpu(), V=V, nd=nd, H=H, L=L, types=names_list), RUN + "s7_events.pt")
print("events", len(keys), "unique", len(uk), "types", V, round(time.time() - t0, 1), "s",
      "peak GPU MB", torch.cuda.max_memory_allocated() // 2**20)
