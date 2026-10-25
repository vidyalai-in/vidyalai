# Load our weights into Hugging Face's LlamaForCausalLM and compare logits.
import torch
from common import heldout_batch
from model import GPT, MAIN
from transformers import LlamaConfig, LlamaForCausalLM

def permute(w, n_heads):
    """Our RoPE pairs (2i, 2i+1) -> Hugging Face's pairs (i, i + head_dim/2), per head (rows of wq/wk)."""
    out, d = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, d).transpose(1, 2).reshape(out, d)

def to_hf(model, rope_permute=True, eps=None):
    c = model.cfg
    hf = LlamaForCausalLM(LlamaConfig(
        vocab_size=c.vocab_size, hidden_size=c.d_model, intermediate_size=c.d_ff,
        num_hidden_layers=c.n_layer, num_attention_heads=c.n_head, num_key_value_heads=c.n_kv_head,
        head_dim=c.head_dim, max_position_embeddings=c.max_seq_len, rms_norm_eps=eps or c.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_base}, tie_word_embeddings=True))
    p = (lambda w, h: permute(w, h)) if rope_permute else (lambda w, h: w)
    sd = {"model.embed_tokens.weight": model.tok_emb.weight, "model.norm.weight": model.norm.weight}
    for i, b in enumerate(model.blocks):
        pre = f"model.layers.{i}."
        sd |= {pre + "input_layernorm.weight": b.attn_norm.weight,
               pre + "self_attn.q_proj.weight": p(b.attn.wq.weight, c.n_head),
               pre + "self_attn.k_proj.weight": p(b.attn.wk.weight, c.n_kv_head),
               pre + "self_attn.v_proj.weight": b.attn.wv.weight,
               pre + "self_attn.o_proj.weight": b.attn.wo.weight,
               pre + "post_attention_layernorm.weight": b.mlp_norm.weight,
               pre + "mlp.gate_proj.weight": b.mlp.gate.weight,
               pre + "mlp.up_proj.weight": b.mlp.up.weight,
               pre + "mlp.down_proj.weight": b.mlp.down.weight}
    missing, unexpected = hf.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    assert missing in ([], ["lm_head.weight"]) and not unexpected, (missing, unexpected)
    return hf.cuda().eval()

@torch.no_grad()
def compare(name, model, hf, x, y):
    for prec in ["fp32", "bf16"]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
            ours, loss = model(x), model(x, y)
            theirs = hf(input_ids=x).logits
        ours, theirs = ours.float(), theirs.float()
        d = (ours - theirs).abs()
        hl = torch.nn.functional.cross_entropy(theirs.view(-1, theirs.size(-1)), y.reshape(-1))
        print(f"{name:34s} {prec}: max |diff| {d.max():.2e}  mean {d.mean():.2e}  logit std {ours.std():.3f}  "
              f"argmax agree {(ours.argmax(-1) == theirs.argmax(-1)).float().mean():.4%}  "
              f"loss ours {loss:.5f} HF {hl:.5f}", flush=True)

x, y = heldout_batch(B=2, T=1024, start=1_000_000)
torch.manual_seed(0)
rand = GPT(MAIN).cuda().eval()
with torch.no_grad():                    # sharpen attention and vary the norm gains, so RoPE and norm bugs show
    for b in rand.blocks:
        b.attn.wq.weight *= 4; b.attn.wk.weight *= 4
    for n, prm in rand.named_parameters():
        if prm.dim() == 1: prm.copy_(1 + 0.2 * torch.randn_like(prm))
trained = GPT(MAIN).cuda().eval()
trained.load_state_dict(torch.load("/home/amit/.cache/vidyalai-scratch/t04/overfit.pt"))
xo, yo = heldout_batch(B=4, T=1024)      # the batch it memorized
hf = to_hf(rand)
print("HF params", f"{sum(p.numel() for p in hf.parameters()):,}", "tied:", hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr())
compare("random (sharpened)", rand, hf, x, y)
compare("random, RoPE not permuted", rand, to_hf(rand, rope_permute=False), x, y)
compare("random, HF default eps 1e-6", rand, to_hf(rand, eps=1e-6), x, y)
hf_t = to_hf(trained)
compare("overfit checkpoint, held-out", trained, hf_t, x, y)
compare("overfit checkpoint, its batch", trained, hf_t, xo[:2], yo[:2])
compare("overfit, RoPE not permuted", trained, to_hf(trained, rope_permute=False), xo[:2], yo[:2])
compare("overfit, HF default eps 1e-6", trained, to_hf(trained, eps=1e-6), x, y)
