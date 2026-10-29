# Reference logits from plain transformers (Gemma 2 2B, bf16, GPU), to check the TL rebuild.
import torch, resource, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
D = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16"
P = ["Fact: the capital of the state containing Dallas is",
     "The Brandenburg Gate is located in the city of",
     "When Mary and John went to the store, John gave a drink to"]
t = time.time()
import glob; tok = AutoTokenizer.from_pretrained(glob.glob("/home/amit/.cache/vidyalai-scratch/hf/hub/models--google--gemma-2-2b/snapshots/*/")[0])
m = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
out = {}
with torch.no_grad():
    for p in P:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        lg = m(ids).logits[0, -1].float()
        top = lg.topk(5)
        out[p] = dict(ids=ids[0].tolist(), logits=lg.cpu(), top=[(tok.decode(i), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
        print(p, out[p]["top"])
torch.save(out, "/home/amit/.cache/vidyalai-scratch/p08/runs/hf_ref.pt")
print(f"{time.time()-t:.0f}s GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB host peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GiB")
