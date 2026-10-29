# Does the exported tokenizer (transformers) match Part 2's tokenizer and Part 10's template?
import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
from transformers import AutoTokenizer
from sft import chat_tokenizer, prompt_ids, render
from evals import fineweb_val_docs
from tokenizer import load_production

out = sys.argv[1]
hf, ours = AutoTokenizer.from_pretrained(out), load_production()
print("bos added:", hf("Hello")["input_ids"], "ours:", ours.encode("Hello").ids)
docs = fineweb_val_docs(n_tokens=300_000)
same = sum(hf(d, add_special_tokens=False)["input_ids"] == ours.encode(d).ids for d in docs)
print(f"plain text: {same}/{len(docs)} documents tokenized identically")
tok = chat_tokenizer()
convs = [[{"role": "user", "content": "What is the capital of France?"}],
         [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": " Give me three tips for sleeping better. "},
          {"role": "assistant", "content": "Sleep at the same time."},
          {"role": "user", "content": "Why?"}]]
for c in convs:
    a = hf.apply_chat_template(c, add_generation_prompt=True, tokenize=True)
    a = a["input_ids"] if isinstance(a, dict) or hasattr(a, "keys") else a
    b = prompt_ids(c, tok)
    print("template match:", a == b, len(b))
print(repr(hf.apply_chat_template(convs[1], add_generation_prompt=True, tokenize=False)))
