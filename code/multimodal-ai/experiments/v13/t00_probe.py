"""Probe: does passing cached encoder outputs (and a reused KV cache) to Qwen2-VL generate give the
same tokens and logits as the plain path?"""
import sys, copy, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import serving as S, vlm
m = vlm.VLM("Qwen/Qwen2-VL-2B-Instruct")
P, M = m.proc, m.model
img = S.report_page(0, 672, 448)
qs = [q for q, a in S.page_questions([0], 3)]
tid = P.tokenizer.convert_tokens_to_ids("<|image_pad|>")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qs[0] + " Answer with the number only."}]}]
pr = P.apply_chat_template(msgs, add_generation_prompt=True)
x = P(text=[pr], images=[img], return_tensors="pt").to("cuda")
n = int((x["input_ids"] == tid).sum())
with torch.inference_mode():
    a = M.generate(**x, max_new_tokens=8, do_sample=False, output_logits=True, return_dict_in_generate=True)
    feats = M.model.get_image_features(pixel_values=x["pixel_values"], image_grid_thw=x["image_grid_thw"], return_dict=True)
    ids = P.tokenizer(pr.replace("<|image_pad|>", "<|image_pad|>" * n), return_tensors="pt")["input_ids"].cuda()
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=x["image_grid_thw"],
              mm_token_type_ids=(ids == tid).long())
    b = M.generate(**kw, mm_encoder_outputs={"image": feats}, max_new_tokens=8, do_sample=False,
                   output_logits=True, return_dict_in_generate=True)
L = x["input_ids"].shape[1]
print("A:", P.decode(a.sequences[0, L:]), "| B:", P.decode(b.sequences[0, L:]))
print("max |logit diff| first token:", (a.logits[0] - b.logits[0]).abs().max().item())
# second turn with KV reuse
rep = P.decode(a.sequences[0, L:], skip_special_tokens=True).strip()
msgs2 = msgs + [{"role": "assistant", "content": [{"type": "text", "text": rep}]},
                {"role": "user", "content": [{"type": "text", "text": qs[1] + " Answer with the number only."}]}]
pr2 = P.apply_chat_template(msgs2, add_generation_prompt=True)
x2 = P(text=[pr2], images=[img], return_tensors="pt").to("cuda")
with torch.inference_mode():
    a2 = M.generate(**x2, max_new_tokens=8, do_sample=False, output_logits=True, return_dict_in_generate=True)
    out1 = M.generate(**x, max_new_tokens=8, do_sample=False, return_dict_in_generate=True)
    cache = out1.past_key_values
    seq1 = out1.sequences[0]
    ids2 = x2["input_ids"]
    k = 0
    while k < min(len(seq1), ids2.shape[1]) and seq1[k] == ids2[0, k]:
        k += 1
    print("cache len", cache.get_seq_length(), "common prefix", k, "new tokens", ids2.shape[1] - k)
    k = min(k, cache.get_seq_length())
    if cache.get_seq_length() > k:
        cache.crop(k - cache.get_seq_length())
    c2 = M.generate(input_ids=ids2, attention_mask=torch.ones_like(ids2), image_grid_thw=x2["image_grid_thw"],
                    mm_token_type_ids=x2["mm_token_type_ids"], past_key_values=cache, max_new_tokens=8,
                    do_sample=False, output_logits=True, return_dict_in_generate=True)
L2 = ids2.shape[1]
print("A2:", P.decode(a2.sequences[0, L2:]), "| C2:", P.decode(c2.sequences[0, L2:]))
print("max |logit diff| first token turn 2:", (a2.logits[0] - c2.logits[0]).abs().max().item(),
      "logit scale", a2.logits[0].abs().max().item())
