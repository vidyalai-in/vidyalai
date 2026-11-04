"""Exp 2: serving a VLM locally (RTX 4060 Laptop GPU, bf16, transformers 5.21).
  stages MODEL  per-stage times: processor (CPU + copy), vision encoder, LM prefill given cached image
                features, whole prefill, decode per token; at 2-3 image sizes. Median of 7 after 2 warm-ups.
  cache         Qwen2-VL-2B, 6-turn conversation about one 1344 x 896 page: (A) every turn from scratch,
                (B) image features cached, (C) features + the KV cache of earlier turns reused.
                1 warm-up conversation + 5 timed; per-turn medians; tokens and logits compared with A.
  batch         Qwen2-VL-2B at 672 x 448: encoder throughput and whole-request throughput by batch size.
Results go to runs/local_<mode>.json."""
import json, sys, time
import torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import serving as S, vlm

D = "/home/amit/.cache/vidyalai-scratch/v13"
QWEN, SMOL = "Qwen/Qwen2-VL-2B-Instruct", "HuggingFaceTB/SmolVLM-256M-Instruct"
mode = sys.argv[1]
SUF = " Answer with the number only."


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup=2, reps=7):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def split(x):
    enc = {k: x[k] for k in ("pixel_values", "image_grid_thw", "pixel_attention_mask") if k in x}
    txt = {k: v for k, v in x.items() if k not in ("pixel_values", "pixel_attention_mask")}
    return enc, txt


def stages(model_id, configs):
    rows = []
    m = None
    for label, (w, h), edge in configs:
        if m is None:   # one Qwen load serves every size; SmolVLM needs a new processor per longest_edge
            m = vlm.VLM(model_id, image_size=edge) if model_id == SMOL else vlm.VLM(model_id)
        M, P = m.model, m.proc
        img = S.report_page(0, w, h)
        q = S.page_questions([0], 1)[0][0] + SUF
        x = m.inputs(img, q)
        enc, txt = split(x)
        M.requires_grad_(False)
        with torch.inference_mode():   # no autograd graph: it would hold every activation alive
            feats = M.model.get_image_features(**enc, return_dict=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = {"model": model_id, "label": label, "w": w, "h": h, "longest_edge": edge,
             "prompt_tokens": int(x["input_ids"].shape[1]), "image_tokens": m.image_tokens(x)}
        with torch.inference_mode(), vlm.GpuSampler() as smi:
            r["proc_ms"] = timed(lambda: m.inputs(img, q, device=None).to("cuda"))
            r["enc_ms"] = timed(lambda: M.model.get_image_features(**enc, return_dict=True))
            r["lm_prefill_ms"] = timed(lambda: M(**txt, mm_encoder_outputs={"image": feats}, logits_to_keep=1))
            r["full_prefill_ms"] = timed(lambda: M(**x, logits_to_keep=1))
            r["gen1_ms"] = timed(lambda: M.generate(**x, max_new_tokens=1, min_new_tokens=1, do_sample=False))
            r["gen33_ms"] = timed(lambda: M.generate(**x, max_new_tokens=33, min_new_tokens=33, do_sample=False))
        r.update(smi.stop())
        r["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        r["decode_ms_per_token"] = (S.med(r["gen33_ms"]) - S.med(r["gen1_ms"])) / 32
        print({k: (round(S.med(v), 2) if isinstance(v, list) else v) for k, v in r.items()}, flush=True)
        rows.append(r)
        if model_id == SMOL:
            del M; m = None; torch.cuda.empty_cache()
    return rows


def conversation(m, img, questions, variant, max_new=12):
    """One multi-turn conversation. Returns per-turn dicts with times (ms), replies and first-token logits."""
    M, P = m.model, m.proc
    tid = P.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    msgs, turns = [], []
    feats = grid = n_img = None
    cache = None
    for t, q in enumerate(questions):
        content = ([{"type": "image"}] if t == 0 else []) + [{"type": "text", "text": q + SUF}]
        msgs.append({"role": "user", "content": content})
        prompt = P.apply_chat_template(msgs, add_generation_prompt=True)
        rec = {"turn": t}
        sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            if variant == "A" or feats is None:
                x = P(text=[prompt], images=[img], return_tensors="pt").to("cuda")
                sync(); t1 = time.perf_counter()
                if variant == "A":
                    t2 = t1
                    gen_kw = dict(x)
                else:
                    enc, txt = split(x)
                    feats = M.model.get_image_features(**enc, return_dict=True)
                    grid = x["image_grid_thw"]
                    n_img = int((x["input_ids"] == tid).sum())
                    sync(); t2 = time.perf_counter()
                    gen_kw = dict(txt, mm_encoder_outputs={"image": feats})
                ids = x["input_ids"]
            else:
                ids = P.tokenizer(prompt.replace("<|image_pad|>", "<|image_pad|>" * n_img), return_tensors="pt")["input_ids"].cuda()
                sync(); t1 = t2 = time.perf_counter()
                gen_kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=grid,
                              mm_token_type_ids=(ids == tid).long())
                if variant == "B":
                    gen_kw["mm_encoder_outputs"] = {"image": feats}
                else:   # C: reuse the KV cache up to the longest common prefix
                    k = 0
                    lim = min(len(cache_seq), ids.shape[1], cache.get_seq_length())
                    while k < lim and cache_seq[k] == ids[0, k]:
                        k += 1
                    if cache.get_seq_length() > k:
                        cache.crop(k - cache.get_seq_length())
                    if k == ids.shape[1]:          # keep at least one token to prefill
                        cache.crop(-1); k -= 1
                    rec["reused_tokens"] = k
                    gen_kw["past_key_values"] = cache
            out = M.generate(**gen_kw, max_new_tokens=max_new, do_sample=False, output_logits=True,
                             return_dict_in_generate=True)
        sync(); t3 = time.perf_counter()
        L = ids.shape[1]
        reply = P.decode(out.sequences[0, L:], skip_special_tokens=True).strip()
        if variant == "C":
            cache, cache_seq = out.past_key_values, out.sequences[0]
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        rec.update(prep_ms=(t1 - t0) * 1000, enc_ms=(t2 - t1) * 1000, gen_ms=(t3 - t2) * 1000,
                   total_ms=(t3 - t0) * 1000, prompt_tokens=L, reply=reply,
                   new_tokens=int(out.sequences.shape[1] - L), logits0=out.logits[0][0].float().cpu())
        turns.append(rec)
    return turns


if mode == "stages":
    which = sys.argv[2]
    if which == "qwen":
        rows = stages(QWEN, [("224", (336, 224), None), ("448", (672, 448), None), ("896", (1344, 896), None)])
    else:
        rows = stages(SMOL, [("default", (1344, 896), None), ("edge512", (1344, 896), 512)])
    json.dump(rows, open(f"{D}/runs/local_stages_{which}.json", "w"), indent=1)

elif mode == "cache":
    m = vlm.VLM(QWEN)
    m.model.requires_grad_(False)
    img = S.report_page(0, 1344, 896)
    qs = [q for q, a in S.page_questions([0], 6, rng_seed=5)]
    answers = [a for q, a in S.page_questions([0], 6, rng_seed=5)]
    res = {"questions": qs, "answers": answers, "runs": {}}
    with vlm.GpuSampler() as smi:
        for rep in range(6):                       # rep 0 is the warm-up
            for v in ("A", "B", "C"):
                turns = conversation(m, img, qs, v)
                res["runs"].setdefault(v, []).append(turns)
                print(rep, v, [round(t["total_ms"], 1) for t in turns], [t["reply"] for t in turns], flush=True)
    res["gpu"] = smi.stop()
    # compare with A (timed reps only)
    cmp = {}
    for v in ("B", "C"):
        diffs, same = [], 0
        for ra, rv in zip(res["runs"]["A"][1:], res["runs"][v][1:]):
            for ta, tv in zip(ra, rv):
                diffs.append(float((ta["logits0"] - tv["logits0"]).abs().max()))
                same += ta["reply"] == tv["reply"]
        cmp[v] = {"max_abs_logit_diff": max(diffs), "median_abs_logit_diff": float(torch.tensor(diffs).median()),
                  "replies_identical": same, "of": len(diffs),
                  "logit_scale": float(res["runs"]["A"][1][0]["logits0"].abs().max())}
    res["compare"] = cmp
    for v in res["runs"]:
        for turns in res["runs"][v]:
            for t in turns:
                t.pop("logits0")
    res["image_tokens"] = int(m.image_tokens(m.inputs(img, qs[0])))
    json.dump(res, open(f"{D}/runs/local_cache.json", "w"), indent=1)
    print(json.dumps(cmp, indent=1))

elif mode == "batch":
    m = vlm.VLM(QWEN)
    M, P = m.model, m.proc
    M.requires_grad_(False)
    P.tokenizer.padding_side = "left"
    pages = [S.report_page(i % 8, 672, 448) for i in range(32)]
    q = S.page_questions([0], 1)[0][0] + SUF
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    pr = P.apply_chat_template(msgs, add_generation_prompt=True)
    rows = []
    for b in (1, 2, 4, 8, 16, 32):
        x = P(text=[pr] * b, images=pages[:b], return_tensors="pt", padding=True).to("cuda")
        enc, _ = split(x)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = {"batch": b, "prompt_tokens": int(x["input_ids"].shape[1])}
        with torch.inference_mode(), vlm.GpuSampler() as smi:
            r["enc_ms"] = timed(lambda: M.model.get_image_features(**enc, return_dict=True), warmup=2, reps=5)
            r["enc_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            r["req_ms"] = timed(lambda: M.generate(**x, max_new_tokens=16, min_new_tokens=16, do_sample=False),
                                warmup=1, reps=5)
            r["req_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        r.update(smi.stop())
        r["images_per_s"] = b / (S.med(r["enc_ms"]) / 1000)
        r["requests_per_s"] = b / (S.med(r["req_ms"]) / 1000)
        r["out_tokens_per_s"] = 16 * r["requests_per_s"]
        print({k: (round(S.med(v), 1) if isinstance(v, list) else v) for k, v in r.items()}, flush=True)
        rows.append(r)
    json.dump(rows, open(f"{D}/runs/local_batch.json", "w"), indent=1)
