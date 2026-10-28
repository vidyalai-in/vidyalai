"""Evaluate one or more models: bits per byte on two texts and multiple-choice tasks. Results -> post/results/<tag>.json."""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import torch
from evals import OurLM, HFLM, fineweb_val_docs, wikitext_docs, bits_per_byte, load_task, score_task, summarize
from sizes import SIZES
from model import MAIN

CFG = {"main": MAIN, **SIZES}
p = argparse.ArgumentParser()
p.add_argument("--model", action="append", required=True, help="tag=ours:<path>:<cfg> or tag=hf:<name>:<fp32|bf16>[:max_len]")
p.add_argument("--bpb", default="fineweb,wikitext")
p.add_argument("--tasks", default="hellaswag,arc_easy,piqa")
p.add_argument("--k", default="0", help="comma-separated shot counts")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--stride", type=int, default=512)
p.add_argument("--suffix", default="")
a = p.parse_args()

texts = {}
tasks = {}
for spec in a.model:
    tag, kind, *rest = spec.replace("=", ":", 1).split(":")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if kind == "ours":
        lm = OurLM(rest[0], cfg=CFG[rest[1]])
    else:
        lm = HFLM(rest[0], bf16=rest[1] == "bf16", max_len=int(rest[2]) if len(rest) > 2 else 1024)
    out = {"spec": spec, "params": sum(q.numel() for q in lm.model.parameters()), "bpb": {}, "tasks": {}, "seconds": {}}
    for name in filter(None, a.bpb.split(",")):
        if name not in texts:
            texts[name] = fineweb_val_docs() if name == "fineweb" else wikitext_docs()
        torch.cuda.synchronize(); t = time.time()
        out["bpb"][name] = bits_per_byte(lm, texts[name], stride=a.stride)
        torch.cuda.synchronize(); out["seconds"]["bpb_" + name] = time.time() - t
        print(tag, name, {k: round(v, 4) for k, v in out["bpb"][name].items()}, f"{time.time() - t:.1f}s", flush=True)
    for name in filter(None, a.tasks.split(",")):
        if name not in tasks:
            tasks[name] = (load_task(name), load_task(name, "train"))
        items, train = tasks[name]
        if a.limit:   # a fixed-seed random subsample
            import random
            idx = sorted(random.Random(0).sample(range(len(items)), a.limit))
            items = [items[i] for i in idx]
        for k in map(int, a.k.split(",")):
            torch.cuda.synchronize(); t = time.time()
            res = score_task(lm, items, k=k, train=train)
            torch.cuda.synchronize(); dt = time.time() - t
            s = summarize(res)
            out["tasks"][f"{name}_{k}"] = {"summary": s, "items": res}
            out["seconds"][f"{name}_{k}"] = dt
            print(tag, name, f"k={k}", {kk: round(v, 4) for kk, v in s.items()}, f"{dt:.1f}s", flush=True)
    out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out["total_seconds"] = time.time() - t0
    json.dump(out, open(f"post/results/{tag}{a.suffix}.json", "w"))
    del lm; torch.cuda.empty_cache()
