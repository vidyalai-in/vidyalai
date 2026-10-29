# Filters for the teacher's replies and the training sets for sequence-level distillation (CPU).
# qwen_filt: teacher replies that pass every filter; qwen_raw: the same number of teacher replies taken in
# order with no filter; orig: the same prompts as qwen_filt with smol-smoltalk's original first reply.
import sys, json, re, collections, statistics
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/fromscratch')
from distill import write_split
from sft import chat_tokenizer
D = '/home/amit/.cache/vidyalai-scratch/t13/data'
P = json.load(open(f'{D}/prompts.json'))
tok = chat_tokenizer()
CJK, IDENT = re.compile(r'[぀-ヿ一-鿿가-힯]'), re.compile(r'Qwen|Alibaba|通义')
def load(split):
    by_i = {p["i"]: p for p in P[split]}
    rows = [json.loads(l) for l in open(f'{D}/gen_{split}.jsonl')]
    rows.sort(key=lambda r: r["i"])
    return [(by_i[r["i"]], r) for r in rows]
def filt(pairs, stats):
    seen, keep = set(), []
    for p, r in pairs:
        t = r["text"].strip()
        why = ("unfinished" if not r["finished"] else "empty" if not t else "cjk" if CJK.search(t)
               else "identity" if IDENT.search(t) else "duplicate" if t in seen else None)
        stats[why or "kept"] += 1
        if why is None:
            seen.add(t); keep.append((p, r))
    return keep
def conv(p, reply): return p["prompt"] + [{"role": "assistant", "content": reply}]
def fits(p, r):                       # both versions fit 1,024 tokens, so every set keeps the same prompts
    n = sum(len(tok.encode(m["content"].strip()).ids) + 2 for m in p["prompt"]) + 3
    return n + len(tok.encode(r["text"].strip()).ids) <= 1024 and n + len(tok.encode(p["reply"].strip()).ids) <= 1024
out = {}
for split in ("train", "val"):
    pairs = load(split)
    st = collections.Counter()
    kept = filt(pairs, st)
    st["too_long_either"] = sum(not fits(p, r) for p, r in kept)
    kept = [(p, r) for p, r in kept if fits(p, r)]
    ours = [len(tok.encode(r["text"].strip()).ids) for _, r in pairs]
    orig = [len(tok.encode(p["reply"].strip()).ids) for p, _ in pairs]
    out[split] = {"generated": len(pairs), "filters": dict(st),
                  "teacher_tokens_total": sum(r["new_tokens"] for _, r in pairs),
                  "teacher_tokens_median": statistics.median(r["new_tokens"] for _, r in pairs),
                  "our_tokens_median_reply": statistics.median(ours), "our_tokens_total": sum(ours),
                  "orig_reply_our_tokens_median": statistics.median(orig), "orig_reply_our_tokens_total": sum(orig),
                  "markdown_share_teacher": sum(bool(re.search(r'\*\*|^#+ |^\d+\. ', r["text"], re.M)) for _, r in pairs) / len(pairs),
                  "markdown_share_orig": sum(bool(re.search(r'\*\*|^#+ |^\d+\. ', p["reply"], re.M)) for p, _ in pairs) / len(pairs)}
    if split == "train":
        n = len(kept)
        out["train"]["kept_qwen_filt"] = write_split([conv(p, r["text"]) for p, r in kept], f'{D}/seq/qwen_filt', "train")
        out["train"]["kept_orig"] = write_split([conv(p, p["reply"]) for p, _ in kept], f'{D}/seq/orig', "train")
        out["train"]["kept_qwen_raw"] = write_split([conv(p, r["text"]) for p, r in pairs[:n]], f'{D}/seq/qwen_raw', "train")
        raw_st = collections.Counter(); filt(pairs[:n], raw_st); out["train"]["raw_subset_would_fail"] = dict(raw_st)
    else:
        out["val"]["evalset_orig"] = write_split([conv(p, p["reply"]) for p, _ in kept], f'{D}/evalsets', "orig")
        out["val"]["evalset_qwen"] = write_split([conv(p, r["text"]) for p, r in kept], f'{D}/evalsets', "qwen")
print(json.dumps(out, indent=1))
json.dump(out, open('/home/amit/.cache/vidyalai-scratch/t13/post/seq_data_stats.json', 'w'), indent=1)
