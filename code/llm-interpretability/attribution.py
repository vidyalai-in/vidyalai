"""Attribution graphs and transcoders for the Interpretability series (owner: Part 8).

Runs in its own venv (/home/amit/.cache/vidyalai-scratch/p08/.venv), because circuit-tracer 0.5.0
pins transformers<=4.57.3 while the shared venv has transformers 5.x:
    circuit-tracer 0.5.0, transformer-lens 3.2.1, transformers 4.57.3, torch 2.11.0+cu128.

Memory plan for an 8 GB GPU and an 8 GB host cap:
- Gemma 2 2B is built as a HookedTransformer directly on the GPU (bf16, no weight processing) and
  filled tensor by tensor from the local bf16 safetensors copy, so no full copy of the model ever
  sits in host RAM (TransformerLens's own loader builds it on the CPU and was OOM-killed at 8 GB).
- Gemma Scope transcoders are converted once from .npz to bf16 .safetensors and loaded lazily:
  encoder and decoder matrices stay on disk and are read only when used.

Functions:
convert_gemma_scope_npz()   one-off .npz -> bf16 safetensors (circuit-tracer's lazy-loading format)
load_transcoders()          TranscoderSet with lazy encoders/decoders
load_gemma_tl()             HookedTransformer-based ReplacementModel for Gemma 2 2B, no host copy
graph_summary()             node/edge counts, replacement and completeness scores, error share
top_nodes()                 the most influential feature nodes of a graph, by layer/position/id
"""
import glob
import json
import os

import einops
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = "/home/amit/.cache/vidyalai-scratch/interp/models/gemma-2-2b-bf16"
TC_DIR = "/home/amit/.cache/vidyalai-scratch/interp/data/p08/gs-transcoders"
# The tokenizer saved next to MODEL_DIR was written by transformers 5 and does not load in 4.57;
# use the cached Hugging Face snapshot directory (a local path, so nothing is fetched).
TOKENIZER_DIR = (glob.glob("/home/amit/.cache/vidyalai-scratch/hf/hub/models--google--gemma-2-2b/snapshots/*/") or [None])[0]


# ---------------------------------------------------------------- transcoders
def convert_gemma_scope_npz(npz_path, out_path, dtype=torch.bfloat16):
    """Gemma Scope stores W_enc as [d_model, d_t]; circuit-tracer wants [d_t, d_model]."""
    p = np.load(npz_path)
    sd = {
        "W_enc": torch.from_numpy(p["W_enc"].T.copy()).to(dtype),
        "W_dec": torch.from_numpy(p["W_dec"]).to(dtype),
        "b_enc": torch.from_numpy(p["b_enc"]).to(dtype),
        "b_dec": torch.from_numpy(p["b_dec"]).to(dtype),
        "activation_function.threshold": torch.from_numpy(p["threshold"]).to(dtype),
    }
    save_file(sd, out_path)
    return {k: tuple(v.shape) for k, v in sd.items()}


def transcoder_paths(tc_dir=TC_DIR):
    paths = {}
    for f in glob.glob(os.path.join(tc_dir, "layer_*_width_16k_*.safetensors")):
        paths[int(os.path.basename(f).split("_")[1])] = f
    return dict(sorted(paths.items()))


def load_transcoders(tc_dir=TC_DIR, device="cuda", dtype=torch.bfloat16, lazy=True):
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
    paths = transcoder_paths(tc_dir)
    return load_transcoder_set(
        paths, scan="gemma-scope-2b-pt-transcoders/width_16k (lowest-L0 set)",
        feature_input_hook="ln2.hook_normalized", feature_output_hook="hook_mlp_out",
        device=torch.device(device), dtype=dtype, special_load_fn=None,
        lazy_encoder=lazy, lazy_decoder=lazy)


# ---------------------------------------------------------------- model
def _gemma_cfg(n_ctx, device, dtype):
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config
    return get_pretrained_model_config("google/gemma-2-2b", device=device, dtype=dtype,
                                       fold_ln=False, n_ctx=n_ctx)


@torch.no_grad()
def _fill_gemma_weights(model, model_dir=MODEL_DIR):
    """Copy Hugging Face Gemma 2 weights into a HookedTransformer, one tensor at a time,
    following transformer_lens.pretrained.weight_conversions.gemma.convert_gemma_weights."""
    cfg = model.cfg
    sd = model.state_dict()
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    done = set()

    def put(name, value):
        sd[name].copy_(value.to(sd[name].dtype))
        done.add(name)

    files = sorted(set(index.values()))
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt", device=str(cfg.device)) as f:
            for key in f.keys():
                if key == "model.embed_tokens.weight":
                    # 1.2 GB: copy in chunks of rows, so no second full copy is ever on the GPU
                    sl, scale = f.get_slice(key), torch.tensor(cfg.d_model ** 0.5, dtype=cfg.dtype)
                    n = sl.get_shape()[0]
                    for i in range(0, n, 16384):
                        w = sl[i:i + 16384].to(cfg.device)
                        sd["embed.W_E"][i:i + 16384].copy_(w * scale)
                        sd["unembed.W_U"][:, i:i + 16384].copy_(w.T)
                    done.update({"embed.W_E", "unembed.W_U"})
                    continue
                w = f.get_tensor(key)
                if key == "model.norm.weight":
                    put("ln_final.w", w.float() + 1)
                    continue
                parts = key.split(".")
                l, rest = int(parts[2]), ".".join(parts[3:])
                b = f"blocks.{l}."
                if rest == "input_layernorm.weight":
                    put(b + "ln1.w", w.float() + 1)
                elif rest == "post_attention_layernorm.weight":
                    put(b + "ln1_post.w", w.float() + 1)
                elif rest == "pre_feedforward_layernorm.weight":
                    put(b + "ln2.w", w.float() + 1)
                elif rest == "post_feedforward_layernorm.weight":
                    put(b + "ln2_post.w", w.float() + 1)
                elif rest == "self_attn.q_proj.weight":
                    put(b + "attn.W_Q", einops.rearrange(w, "(n h) m -> n m h", n=cfg.n_heads))
                elif rest == "self_attn.k_proj.weight":
                    put(b + "attn._W_K", einops.rearrange(w, "(n h) m -> n m h", n=cfg.n_key_value_heads))
                elif rest == "self_attn.v_proj.weight":
                    put(b + "attn._W_V", einops.rearrange(w, "(n h) m -> n m h", n=cfg.n_key_value_heads))
                elif rest == "self_attn.o_proj.weight":
                    put(b + "attn.W_O", einops.rearrange(w, "m (n h) -> n h m", n=cfg.n_heads))
                elif rest == "mlp.up_proj.weight":
                    put(b + "mlp.W_in", w.T)
                elif rest == "mlp.gate_proj.weight":
                    put(b + "mlp.W_gate", w.T)
                elif rest == "mlp.down_proj.weight":
                    put(b + "mlp.W_out", w.T)
                else:
                    raise KeyError(key)
                del w
    for name, t in sd.items():          # Gemma has no biases
        if name.split(".")[-1].startswith(("b_", "_b_")) and name not in done:
            t.zero_()
            done.add(name)
    missing = [n for n in sd if n not in done and not n.endswith(("mask", "IGNORE", "rotary_sin", "rotary_cos"))]
    assert not missing, missing


def load_gemma_tl(transcoders=None, n_ctx=256, device="cuda", dtype=torch.bfloat16,
                  model_dir=MODEL_DIR):
    """Gemma 2 2B as a TransformerLens model on the GPU, without weight processing
    (no LayerNorm folding, no centering). With transcoders, returns circuit-tracer's
    TransformerLensReplacementModel; without, a plain HookedTransformer."""
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    cfg = _gemma_cfg(n_ctx, device, dtype)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if transcoders is not None:
        from circuit_tracer.replacement_model.replacement_model_transformerlens import (
            TransformerLensReplacementModel as cls)
    else:
        cls = HookedTransformer
    with torch.device(device):
        model = cls(cfg, tokenizer=tok)
    _fill_gemma_weights(model, model_dir)
    model.eval()
    if transcoders is not None:
        model._configure_replacement_model(transcoders)
    return model


# ---------------------------------------------------------------- graphs
def graph_summary(graph, node_threshold=0.8, edge_threshold=0.98):
    from circuit_tracer.graph import compute_graph_scores, prune_graph
    n_feat = len(graph.selected_features)
    n_tok = len(graph.input_tokens)
    n_layers = graph.cfg.n_layers
    n_err = n_tok * n_layers
    rep, comp = compute_graph_scores(graph)
    pr = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    nm = pr.node_mask
    kept_feat = int(nm[:n_feat].sum())
    kept_err = int(nm[n_feat:n_feat + n_err].sum())
    return dict(
        prompt=graph.input_string, n_tokens=n_tok,
        active_features=int(graph.active_features.shape[0]),
        feature_nodes=n_feat, error_nodes=n_err, logit_nodes=len(graph.logit_targets),
        replacement_score=rep, completeness_score=comp,
        pruned_feature_nodes=kept_feat, pruned_error_nodes=kept_err,
        pruned_edges=int(pr.edge_mask.sum()),
        node_threshold=node_threshold, edge_threshold=edge_threshold)


def top_nodes(graph, k=20):
    """Feature nodes ranked by influence on the logits: (layer, pos, feature_id, influence, act)."""
    from circuit_tracer.graph import compute_influence, normalize_matrix
    n_logits = len(graph.logit_targets)
    lw = torch.zeros(graph.adjacency_matrix.shape[0])
    lw[-n_logits:] = graph.logit_probabilities.cpu()
    infl = compute_influence(normalize_matrix(graph.adjacency_matrix.cpu()), lw)
    n_feat = len(graph.selected_features)
    order = infl[:n_feat].argsort(descending=True)[:k]
    out = []
    for i in order.tolist():
        layer, pos, fid = graph.active_features[graph.selected_features[i]].tolist()
        act = graph.activation_values[graph.selected_features[i]].item()
        out.append(dict(layer=layer, pos=pos, feature=fid, influence=infl[i].item(), act=act))
    return out
