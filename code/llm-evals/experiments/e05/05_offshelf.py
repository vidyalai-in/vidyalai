"""Detectors on off-the-shelf models (GPU; run under the lock), Experiment 2.
  python 05_offshelf.py items <name>    # verbatim ARC-Challenge items vs their Haiku paraphrases
  python 05_offshelf.py shard <name> <bench>  # exchangeability test on a public test set
<name>: qwen05 | smol135 | pythia160 | ours88.  <bench>: arc | mmlu | gsm8k
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import contamination as C

RUNS = "/home/amit/.cache/vidyalai-scratch/e05/runs"
HF = {"qwen05": "Qwen/Qwen2.5-0.5B", "smol135": "HuggingFaceTB/SmolLM2-135M", "pythia160": "EleutherAI/pythia-160m"}


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class Ours(torch.nn.Module):
    """From Scratch Part 7's 88M model behind the two calls contamination.py uses."""

    def __init__(self):
        super().__init__()
        sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
        from model import MAIN
        from snapshots import load
        self.m = load("/home/amit/.cache/vidyalai-scratch/t07/runs/main88/model_03360.pt", cfg=MAIN, device="cuda")
        self.m.eval()

    def base_model(self, input_ids, attention_mask=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.m.tok_emb(input_ids)
            for b in self.m.blocks:
                x = b(x, self.m.rope_cos, self.m.rope_sin)
            return _Out(self.m.norm(x).float())

    def get_output_embeddings(self):
        return lambda h: F.linear(h, self.m.tok_emb.weight.float())


class OursTok:
    bos_token_id, pad_token_id = None, 0

    def __init__(self):
        sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
        from tokenizer import load_production
        self.t = load_production()
        self.t.encode_special_tokens = True

    def __call__(self, text, add_special_tokens=False):
        return _Ids(self.t.encode(text).ids)


class _Ids:
    def __init__(self, ids):
        self.input_ids = ids


def load(name):
    if name == "ours88":
        return Ours(), OursTok()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[name])
    m = AutoModelForCausalLM.from_pretrained(HF[name], dtype=torch.bfloat16, device_map="cuda").eval()
    if name == "pythia160":
        m = m.float()  # Pythia's scores degrade under bf16 (From Scratch Part 9)
    return m, tok


def arc_items():
    return json.load(open("/home/amit/.cache/vidyalai-scratch/evals/data/e01-arc-challenge.json"))


def qa(q, a):
    return f"Question: {q}\nAnswer: {a}"


t0 = time.time()
mode, name = sys.argv[1], sys.argv[2]
model, tok = load(name)
if mode == "items":
    items = arc_items()
    para = {int(json.loads(l)["id"][1:]): json.loads(l)["text"].strip() for l in open(f"{RUNS}/paraphrase-clean.jsonl")}
    ids = [i for i in range(len(items)) if i in para]
    out = {"ids": ids}
    for kind in ("verbatim", "para"):
        texts = [qa(items[i]["question"] if kind == "verbatim" else para[i], items[i]["choices"][items[i]["answer"]]) for i in ids]
        st = C.token_stats(model, tok, texts, batch_size=32, add_bos=False)
        out[kind] = [C.detectors(s, t) for s, t in zip(st, texts)]
    json.dump(out, open(f"{RUNS}/offshelf-items-{name}.json", "w"))
else:
    bench = sys.argv[3]
    if bench == "arc":
        rows = [qa(it["question"], it["choices"][it["answer"]]) for it in arc_items()]
    elif bench == "mmlu":
        import pandas as pd, glob
        f = sorted(glob.glob("/home/amit/.cache/vidyalai-scratch/evals/data/e05-mmlu-test.parquet"))[0]
        df = pd.read_parquet(f).iloc[:1170]
        # the original release's CSV row: question, four options, answer letter
        rows = [",".join([r.question, *r.choices, "ABCD"[r.answer]]) for r in df.itertuples()]
    elif bench == "gsm8k":
        rows = [json.dumps({"question": json.loads(l)["question"], "answer": json.loads(l)["answer"]})
                for l in open("/home/amit/.cache/vidyalai-scratch/evals/data/e05-gsm8k-test.jsonl")][:450]
    per = 9 if bench != "gsm8k" else 3
    shards = [rows[k:k + per] for k in range(0, len(rows) - per + 1, per)]
    sep = "\n" if bench in ("mmlu", "gsm8k") else "\n\n"
    r = C.shard_test(model, tok, shards, n_perm=int(sys.argv[4]) if len(sys.argv) > 4 else 50, seed=0, sep=sep, batch_size=4)
    r.update(model=name, bench=bench, n_examples=sum(len(s) for s in shards), per_shard=per)
    json.dump(r, open(f"{RUNS}/offshelf-shard-{name}-{bench}.json", "w"))
    print(name, bench, "p", r["p"], "t", r["t"], "mean diff", r["mean_diff"])
print("done", round(time.time() - t0), "s", "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
