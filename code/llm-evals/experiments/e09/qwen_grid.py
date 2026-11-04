"""The same tasks on an open model with a 32K native window, with local prefix caching:
each haystack is prefilled once and its KV cache is copied for every question.
usage: qwen_grid.py <hf model id> <tag>"""
import sys, json, time, copy
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/evals/lab")
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import longctx as lc

mid, tag = sys.argv[1], sys.argv[2]
OUT = f"/home/amit/.cache/vidyalai-scratch/e09/runs/open-{tag}.jsonl"
LENGTHS = [0, 4000, 8000, 16000, 31500]   # Qwen tokens in the haystack prefix
H, K = 4, 6
MAXNEW = {"literal": 24, "onehop": 24, "vt": 128, "know": 16}
tok = AutoTokenizer.from_pretrained(mid)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()
cfg = model.config
print(mid, "loaded in %.1f s" % (time.time() - t0), "max_position_embeddings", cfg.max_position_embeddings,
      "layers", cfg.num_hidden_layers, "kv heads", cfg.num_key_value_heads, "head dim", cfg.hidden_size // cfg.num_attention_heads,
      "torch", torch.__version__, "transformers", transformers.__version__, flush=True)
paras = lc.load_corpus()
cpt = len(lc.filler(paras, 50000, 3000)) / len(tok(lc.filler(paras, 50000, 3000)).input_ids)
print("chars per Qwen token %.3f" % cpt)
f = open(OUT, "w")


@torch.no_grad()
def answer(pre_cache, pre_ids, suf_ids, max_new):
    ids = torch.tensor([pre_ids + suf_ids], device="cuda")
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=copy.deepcopy(pre_cache),
                         max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# world-knowledge control
for i, (lm, city) in enumerate(lc.LANDMARKS):
    ids = tok.apply_chat_template([{"role": "user", "content": lc.Q_KNOW.format(landmark=lm)}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=16, do_sample=False,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    f.write(json.dumps(dict(task="know", qid=f"know-{i}", gold=city, reply=reply, correct=lc.grade_city(reply, city))) + "\n")

for li, L in enumerate([0, 4000, 8000, 16000, 32000, 64000, 96000][:5]):
    Lq = LENGTHS[li]
    for h in range(H):
        depths = [(j * H + h) / (K * H - 1) for j in range(K)]
        seed, start = 1000 + 10 * li + h, (li * H + h) * 389 % 10000   # same needles and filler start as the Claude grid
        n = int((Lq - 450) * cpt) if Lq else 0
        hay = lc.build_haystack(paras, n, start, seed, k=K, char_depths=depths)
        qs = lc.questions(hay)
        pre, _ = lc.hf_prompt_parts(tok, hay["text"], qs[0][2])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t1 = time.time()
        with torch.no_grad():
            o = model(input_ids=torch.tensor([pre], device="cuda"), use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(); t_pre = time.time() - t1
        cache = o.past_key_values
        kv_bytes = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
        peak_prefill = torch.cuda.max_memory_allocated()
        t2 = time.time()
        for qid, task, q, gold, depth in qs:
            pre2, suf = lc.hf_prompt_parts(tok, hay["text"], q)
            assert pre2 == pre
            reply = answer(cache, pre, suf, MAXNEW[task])
            rec = dict(model=mid, L=L, Lq=Lq, h=h, qid=qid, task=task, gold=gold, depth=depth, reply=reply,
                       n_tokens=len(pre) + len(suf))
            if task == "vt":
                rec["correct"], rec["recall"] = lc.grade_vt(reply, gold)
            else:
                rec["correct"] = lc.grade_name(reply, gold)
                rec["hedged"] = lc.hedged(reply)
            f.write(json.dumps(rec) + "\n")
        t_q = time.time() - t2
        f.write(json.dumps(dict(event="haystack", model=mid, L=L, Lq=Lq, h=h, prefix_tokens=len(pre), prefill_s=round(t_pre, 3),
                                kv_cache_bytes=kv_bytes, peak_prefill_bytes=peak_prefill,
                                peak_all_bytes=torch.cuda.max_memory_allocated(), questions_s=round(t_q, 2))) + "\n")
        f.flush()
        print(L, h, len(pre), "prefill %.2fs" % t_pre, "KV %.0f MiB" % (kv_bytes / 2**20),
              "peak %.0f MiB" % (torch.cuda.max_memory_allocated() / 2**20), "15 questions %.1fs" % t_q, flush=True)
        del cache, o
        torch.cuda.empty_cache()
print("total %.1f s" % (time.time() - t0))
