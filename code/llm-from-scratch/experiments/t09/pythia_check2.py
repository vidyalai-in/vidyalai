import torch, sys, torch.nn.functional as F
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from evals import fineweb_val_docs
from transformers import AutoModelForCausalLM, AutoTokenizer
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
docs = fineweb_val_docs(30000)
for bf in (False, True):
    tot, n = 0, 0
    for d in docs[:12]:
        ids = torch.tensor([[tok.eos_token_id] + tok(d)["input_ids"]][:1])[:, :1025]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf):
            lg = m(ids).logits
        lp = F.log_softmax(lg.float(), -1)[0, :-1].gather(-1, ids[0, 1:, None])
        tot += -lp.sum().item(); n += ids.shape[1] - 1
    print(name, "bf16" if bf else "fp32", tot / n, n)
