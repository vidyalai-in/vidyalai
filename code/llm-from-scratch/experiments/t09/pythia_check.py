import torch, sys, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_info()
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
print(m.config)
text = ("The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, "
        "cools and condenses into clouds, and falls again to the surface as precipitation. This process is driven by energy from the sun.")
ids = torch.tensor([[tok.eos_token_id] + tok(text)["input_ids"]])
with torch.no_grad():
    lp = F.log_softmax(m(ids).logits.float(), -1)[0, :-1].gather(-1, ids[0, 1:, None])
print(name, "fp32 loss", -lp.mean().item(), "tokens", ids.shape[1] - 1)
