"""Exporting our model as a Hugging Face Llama checkpoint (Build an LLM From Scratch, Part 15).

    python export.py runs/main88/model_03360.pt export/base88                  # base model
    python export.py runs/onpolicy/model_final.pt export/chat88 --chat         # chat model
    python export.py ... --check                                               # compare logits too

The output folder holds config.json, model.safetensors, tokenizer.json, tokenizer_config.json
and generation_config.json: what transformers' LlamaForCausalLM and llama.cpp's
convert_hf_to_gguf.py expect. Weights are copied as they are, except the query and key rows,
which are reordered for Hugging Face's RoPE layout (Part 4).
"""
import argparse
import json
import os
import shutil

import torch

from model import GPT, MAIN
from tokenizer import SPECIAL_TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))

# Part 10's template: <|endoftext|>, then <|role|> text <|end|> per message; the reply follows <|assistant|>.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + (message['content'] | trim) + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def permute(w, n_heads):
    """Our RoPE pairs (2i, 2i+1) -> Hugging Face's pairs (i, i + head_dim/2), per head (rows of wq/wk)."""
    out, d = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, d).transpose(1, 2).reshape(out, d)


def hf_config(cfg, chat):
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layer, num_attention_heads=cfg.n_head, num_key_value_heads=cfg.n_kv_head,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,                                   # 1e-5; Hugging Face's default is 1e-6
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_base},
        hidden_act="silu", attention_bias=False, mlp_bias=False, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=4 if chat else 0, pad_token_id=0, torch_dtype="float32")


def hf_state_dict(model):
    """Our parameter names -> LlamaForCausalLM's. lm_head is absent: it is the tied embedding."""
    c, sd = model.cfg, {"model.embed_tokens.weight": model.tok_emb.weight, "model.norm.weight": model.norm.weight}
    for i, b in enumerate(model.blocks):
        p = f"model.layers.{i}."
        sd |= {p + "input_layernorm.weight": b.attn_norm.weight,
               p + "self_attn.q_proj.weight": permute(b.attn.wq.weight, c.n_head),
               p + "self_attn.k_proj.weight": permute(b.attn.wk.weight, c.n_kv_head),
               p + "self_attn.v_proj.weight": b.attn.wv.weight,
               p + "self_attn.o_proj.weight": b.attn.wo.weight,
               p + "post_attention_layernorm.weight": b.mlp_norm.weight,
               p + "mlp.gate_proj.weight": b.mlp.gate.weight,
               p + "mlp.up_proj.weight": b.mlp.up.weight,
               p + "mlp.down_proj.weight": b.mlp.down.weight}
    return {k: v.detach().float().contiguous().clone() for k, v in sd.items()}


def write_tokenizer(out, chat):
    """Part 2's tokenizer.json unchanged, plus the settings transformers and llama.cpp read."""
    shutil.copy(os.path.join(HERE, "tokenizer-32k.json"), os.path.join(out, "tokenizer.json"))
    conf = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": MAIN.max_seq_len,
            "bos_token": "<|endoftext|>", "eos_token": "<|end|>" if chat else "<|endoftext|>",
            "pad_token": "<|endoftext|>", "add_bos_token": True, "add_eos_token": False,
            "clean_up_tokenization_spaces": False, "additional_special_tokens": SPECIAL_TOKENS[1:]}
    if chat:
        conf["chat_template"] = CHAT_TEMPLATE
    json.dump(conf, open(os.path.join(out, "tokenizer_config.json"), "w"), indent=1)


def export(ckpt, out, chat=False):
    from transformers import GenerationConfig, LlamaForCausalLM
    from snapshots import load
    model = load(ckpt, device="cpu")
    hf = LlamaForCausalLM(hf_config(model.cfg, chat))
    missing, unexpected = hf.load_state_dict(hf_state_dict(model), strict=False)
    assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()   # still tied
    os.makedirs(out, exist_ok=True)
    hf.save_pretrained(out)
    GenerationConfig(bos_token_id=0, eos_token_id=[4, 0] if chat else 0, pad_token_id=0).save_pretrained(out)
    write_tokenizer(out, chat)
    return model, hf


@torch.no_grad()
def check(model, out, texts, device="cuda"):
    """Max |logit difference| between our model and the saved checkpoint, reloaded, in FP32."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)
    hf = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).to(device).eval()
    model = model.to(device).float()
    worst = 0.0
    for text in texts:
        ids = torch.tensor([tok(text)["input_ids"][:model.cfg.max_seq_len]], device=device)
        ours, theirs = model(ids), hf(input_ids=ids).logits
        worst = max(worst, (ours - theirs).abs().max().item())
        same = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
        print(f"{ids.shape[1]:5d} tokens  max |diff| {(ours - theirs).abs().max().item():.2e}  "
              f"argmax agree {same:.2%}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--chat", action="store_true", help="eos <|end|> and Part 10's chat template")
    ap.add_argument("--check", action="store_true", help="compare logits on held-out text (GPU)")
    a = ap.parse_args()
    model, _ = export(a.ckpt, a.out, a.chat)
    print("wrote", a.out, sorted(os.listdir(a.out)))
    if a.check:
        from evals import fineweb_val_docs
        docs = fineweb_val_docs(n_tokens=20_000)[:4]
        print("worst max |diff|:", check(model, a.out, docs))
