# Compression of our tokenizers (4K-64K) and real tokenizers on held-out text.
import sys, json, glob, os, re, time
import numpy as np
from tokenizers import Tokenizer
import tiktoken
FS = "/home/amit/.cache/vidyalai-scratch/fromscratch"; HUB = "/home/amit/.cache/vidyalai-scratch/hf/hub"
held = [json.loads(l)["text"] for l in open(f"{FS}/data/tokenizer-sample/heldout.jsonl")]
code = [open(f, encoding="utf-8").read() for f in sorted(glob.glob("/usr/lib/python3.12/*.py"))]
nums = [" ".join(re.findall(r"\d[\d,.]*\d|\d", t)) for t in held]
nums = [s for s in nums if s]
FL = "/home/amit/.cache/vidyalai-scratch/dl/flores/flores200_dataset"
def flores(l): return [x.rstrip("\n") for s in ["dev", "devtest"] for x in open(f"{FL}/{s}/{l}.{s}", encoding="utf-8")]
hin, eng = flores("hin_Deva"), flores("eng_Latn")
sets = {"fineweb_heldout": held, "python_code": code, "numbers": nums, "flores_eng": eng, "flores_hin": hin}
def hub(repo): return Tokenizer.from_file(glob.glob(f"{HUB}/models--{repo}/snapshots/*/tokenizer.json")[0])
toks = {}
for V in [4096, 8192, 16384, 32768, 65536]:
    p = f"{FS}/tokenizer-32k.json" if V == 32768 else f"/home/amit/.cache/vidyalai-scratch/t02/tok/tokenizer-{V}.json"
    toks[f"ours-{V}"] = ("hf", Tokenizer.from_file(p))
toks["gpt2"] = ("tt", tiktoken.get_encoding("gpt2"))
toks["llama2"] = ("hf", hub("NousResearch--Llama-2-7b-hf"))
toks["smollm2"] = ("hf", hub("HuggingFaceTB--SmolLM2-135M"))
toks["cl100k"] = ("tt", tiktoken.get_encoding("cl100k_base"))
toks["llama3"] = ("hf", hub("NousResearch--Meta-Llama-3-8B"))
toks["qwen2.5"] = ("hf", hub("Qwen--Qwen2.5-0.5B"))
toks["o200k"] = ("tt", tiktoken.get_encoding("o200k_base"))
def encode(kind, t, docs):
    if kind == "tt": return t.encode_ordinary_batch(docs)
    return [e.ids for e in t.encode_batch(docs, add_special_tokens=False)]
res = {}
for name, (kind, t) in toks.items():
    r = {"vocab": t.n_vocab if kind == "tt" else t.get_vocab_size()}
    for sname, docs in sets.items():
        if name.startswith("ours-") and name != "ours-32768" and sname != "fineweb_heldout": continue
        t0 = time.perf_counter(); ids = encode(kind, t, docs); dt = time.perf_counter() - t0
        ntok = sum(map(len, ids)); nbytes = sum(len(d.encode()) for d in docs)
        r[sname] = {"tokens": ntok, "bytes": nbytes, "bytes_per_token": nbytes / ntok,
                    "chars_per_token": sum(map(len, docs)) / ntok, "tokens_per_word": ntok / sum(len(d.split()) for d in docs), "s": round(dt, 2)}
        if sname == "fineweb_heldout":
            used = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in ids if x]))
            r[sname]["vocab_used_frac"] = len(used) / r["vocab"]
    res[name] = r
    print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
json.dump(res, open("/home/amit/.cache/vidyalai-scratch/t02/eval_compare.json", "w"), indent=1)
