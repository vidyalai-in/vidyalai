"""aMUSEd-256 text-to-image on the laptop GPU: CLIP score vs decoding steps, and time per image.
Usage: python amused_run.py gen|time"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/v08")
import torch
from diffusers import AmusedPipeline
from transformers import CLIPModel, CLIPProcessor

import ar_image as ai
from prompts import PROMPTS

OUT = Path("/home/amit/.cache/vidyalai-scratch/v08/amused")
OUT.mkdir(exist_ok=True)
STEPS = [1, 2, 4, 8, 12, 16, 24]
SEEDS = [0, 1, 2]


def _feat(out):
    # transformers 5 may return a ModelOutput whose pooler_output is the projected embedding (as in lab/clip_utils.py)
    if isinstance(out, torch.Tensor):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, k, None) is not None:
            return getattr(out, k)
    raise TypeError(type(out))


def load():
    pipe = AmusedPipeline.from_pretrained("amused/amused-256", variant="fp16", torch_dtype=torch.float16)
    pipe.vqvae.to(torch.float32)  # the model card: "vqvae is producing nans in fp16"
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen():
    pipe = load()
    configs = [(s, 10.0) for s in STEPS] + [(12, 1.0)]
    meta = []
    t0 = time.time()
    for steps, g in configs:
        for seed in SEEDS:
            gen_ = torch.Generator("cuda").manual_seed(seed)
            imgs = []
            for j in range(0, len(PROMPTS), 20):  # two calls of 20 prompts, one generator per (setting, seed)
                imgs += pipe(PROMPTS[j:j + 20], num_inference_steps=steps, guidance_scale=g, generator=gen_).images
            for i, im in enumerate(imgs):
                f = OUT / f"s{steps:02d}_g{int(g)}_seed{seed}_p{i:02d}.png"
                im.save(f)
                meta.append({"steps": steps, "guidance": g, "seed": seed, "prompt_id": i, "file": f.name})
        print(steps, g, round(time.time() - t0, 1), flush=True)
    del pipe
    torch.cuda.empty_cache()
    # CLIP score with a different CLIP than aMUSEd's own text encoder (CLIP-L/14): ViT-B/32
    from PIL import Image
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        t = proc(text=PROMPTS, return_tensors="pt", padding=True).to("cuda")
        te = _feat(clip.get_text_features(**t))
        te = te / te.norm(dim=-1, keepdim=True)
        for i in range(0, len(meta), 64):
            chunk = meta[i:i + 64]
            ims = [Image.open(OUT / m["file"]).convert("RGB") for m in chunk]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
            ie = _feat(clip.get_image_features(pixel_values=px))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            sims = (ie @ te.t()).float().cpu()
            for m, row in zip(chunk, sims):
                m["clip"] = 100 * row[m["prompt_id"]].item()
                others = torch.cat([row[:m["prompt_id"]], row[m["prompt_id"] + 1:]])
                m["clip_mismatch"] = 100 * others.mean().item()
                m["clip_rank"] = int((row > row[m["prompt_id"]]).sum().item()) + 1
    (OUT.parent / "runs").mkdir(exist_ok=True)
    with open(OUT.parent / "runs" / "amused_clip.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print("done", len(meta), round(time.time() - t0, 1))


def timing():
    pipe = load()
    p = "a red fox in the snow"
    res = {}
    for steps in STEPS:
        r = ai.time_cuda(lambda: pipe(p, num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(0)),
                         runs=7, warmup=3)
        res[f"steps{steps}_g10"] = r
        print(steps, r, flush=True)
    r = ai.time_cuda(lambda: pipe(p, num_inference_steps=12, guidance_scale=1.0,
                                  generator=torch.Generator("cuda").manual_seed(0)), runs=7, warmup=3)
    res["steps12_g1"] = r
    print("g1", r)
    try:
        # components at batch 1 (2 with guidance)
        tok = pipe.tokenizer([p, ""], return_tensors="pt", padding="max_length", truncation=True,
                             max_length=pipe.tokenizer.model_max_length).input_ids.to("cuda")
        with torch.no_grad():
            res["text_encoder"] = ai.time_cuda(lambda: pipe.text_encoder(tok, return_dict=True, output_hidden_states=True), 9, 3)
            out = pipe.text_encoder(tok, return_dict=True, output_hidden_states=True)
            ehs = out.hidden_states[-2]
            pooled = out.text_embeds
            lat = torch.full((2, 16, 16), pipe.scheduler.config.mask_token_id, device="cuda", dtype=torch.long)
            mc = torch.tensor([256, 256, 0, 0, 6], device="cuda", dtype=torch.float16)[None].expand(2, -1)
            res["transformer_pass_b2"] = ai.time_cuda(
                lambda: pipe.transformer(lat, micro_conds=mc, pooled_text_emb=pooled, encoder_hidden_states=ehs), 9, 3)
            codes = torch.randint(0, 8192, (1, 16, 16), device="cuda")
            res["vq_decode"] = ai.time_cuda(lambda: pipe.vqvae.decode(codes, force_not_quantize=True,
                                                                      shape=(1, 16, 16, pipe.vqvae.config.latent_channels)), 9, 3)
            # batch-8 throughput at the default 12 steps
            res["steps12_g10_batch8"] = ai.time_cuda(lambda: pipe([p] * 8, num_inference_steps=12,
                                                                   generator=torch.Generator("cuda").manual_seed(0)), 5, 2)
    except Exception as e:
        print('component timing failed', repr(e))
    res["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["params"] = {k: sum(x.numel() for x in getattr(pipe, k).parameters()) for k in ("transformer", "text_encoder", "vqvae")}
    res["torch"] = torch.__version__
    (OUT.parent / "runs" / "amused_timing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["params"]), res["max_mem_gb"])


if __name__ == "__main__":
    {"gen": gen, "time": timing}[sys.argv[1]]()
