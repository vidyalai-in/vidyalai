# Compare prompt templates and BOS on the fact table (GPT-2 small, fp32, processed weights).
import sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp"); sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/p01/scripts")
from lab import hooks as H
from facts import FACTS
torch.set_grad_enabled(False)
m = H.load("gpt2")
facts = [f for f in FACTS if f[1] not in ("Agra", "Granada", "Krakow", "Pisa", "Mecca")]
cities = sorted({c for _, c in facts})
cid = torch.tensor([m.to_tokens(" " + c, prepend_bos=False)[0].item() for c in cities], device="cuda")
ans = torch.tensor([cities.index(c) for _, c in facts], device="cuda")
for tpl in ["{} is located in the city of", "{} is in the city of", "{} is a famous landmark in the city of", "The city where {} is located is"]:
    for bos in (True, False):
        top1 = topc = 0
        for (l, c), a in zip(facts, ans):
            lg = m(m.to_tokens(tpl.format(l), prepend_bos=bos))[0, -1]
            top1 += int(lg.argmax() == cid[a]); topc += int(lg[cid].argmax() == a)
        print(f"{tpl!r:48s} bos={bos!s:5s} top1 {top1/len(facts):.3f} among-cities {topc/len(facts):.3f}")
lg = m(m.to_tokens("The Eiffel Tower is located in the city of", prepend_bos=False))[0, -1]
print("Eiffel no BOS top:", [m.to_string(t) for t in lg.topk(3).indices])
