"""(b) The local token ledger: exact shapes each small model produces per image, per second of audio and
per second of video, with GPU timings (median of >= 10 runs after 3 warm-ups, nvidia-smi sampled).
Usage: python local_ledger.py <section> ...   sections: vit vae vq whisper mimi procs vivit
Each section appends its results to runs/local_<section>.json.
"""
import json, math, sys, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import tokens

OUT = "/home/amit/.cache/vidyalai-scratch/v01/runs"
DEV, DT = "cuda", torch.bfloat16
IMG = tokens.scene(512, 512)                       # the test picture, 512 x 512 RGB
RAW_IMG = 512 * 512 * 3                            # 786,432 values of 8 bits


def to_tensor(img):                                # [-1, 1], NCHW
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().div(127.5).sub(1)


def psnr(a, b):                                    # images in [-1, 1]
    mse = torch.mean(((a.float().clamp(-1, 1) - b.float()) / 2) ** 2).item()
    return 10 * math.log10(1 / mse)


def save(name, obj):
    json.dump(obj, open(f"{OUT}/local_{name}.json", "w"), indent=1)
    print(name, json.dumps(obj)[:1500], flush=True)


@torch.no_grad()
def vit():
    from transformers import ViTModel, ViTImageProcessor
    mid = "google/vit-base-patch16-224"
    proc = ViTImageProcessor.from_pretrained(mid)
    model = ViTModel.from_pretrained(mid, dtype=DT, device_map=DEV).eval()
    x = proc(images=IMG, return_tensors="pt")["pixel_values"].to(DEV, DT)
    out = model(pixel_values=x).last_hidden_state
    frames = torch.cat([proc(images=f, return_tensors="pt")["pixel_values"] for f in tokens.video(1.0, 24, 512, 512)]).to(DEV, DT)
    save("vit", {"model": mid, "input_after_processor": list(x.shape), "output": list(out.shape),
                 "time_image": tokens.time_gpu(lambda: model(pixel_values=x)),
                 "time_24_frames": tokens.time_gpu(lambda: model(pixel_values=frames))})


@torch.no_grad()
def vae():
    from diffusers import AutoencoderKL
    mid = "stabilityai/sd-vae-ft-mse"
    model = AutoencoderKL.from_pretrained(mid, torch_dtype=DT).to(DEV).eval()
    x = to_tensor(IMG).to(DEV, DT)
    z = model.encode(x).latent_dist.mean
    rec = model.decode(z).sample
    frames = torch.cat([to_tensor(f) for f in tokens.video(1.0, 24, 512, 512)[:8]]).to(DEV, DT)
    save("vae", {"model": mid, "input": list(x.shape), "latent": list(z.shape), "psnr_db": psnr(rec.cpu(), x.cpu()),
                 "latent_mean": z.float().mean().item(), "latent_std": z.float().std().item(),
                 "time_encode_image": tokens.time_gpu(lambda: model.encode(x)),
                 "time_encode_8_frames": tokens.time_gpu(lambda: model.encode(frames), runs=5)})


@torch.no_grad()
def vq():
    from diffusers import VQModel
    mid = "amused/amused-256"
    try:
        model = VQModel.from_pretrained(mid, subfolder="vqvae", variant="fp16", torch_dtype=DT).to(DEV).eval()
        dt = DT
    except Exception as e:
        print("bf16 load failed", e)
        raise
    res = {}
    for size in (256, 512):
        x = to_tensor(tokens.scene(size, size)).to(DEV, dt)
        h = model.encode(x).latents
        quant, _, info = model.quantize(h)
        idx = info[2]
        rec = model.decode(idx.reshape(1, *h.shape[-2:]), force_not_quantize=True, shape=(1, *h.shape[-2:], h.shape[1])).sample
        if torch.isnan(rec).any():
            dt = torch.float32
            model = model.to(dt)
            x = x.to(dt); h = model.encode(x).latents; quant, _, info = model.quantize(h); idx = info[2]
            rec = model.decode(idx.reshape(1, *h.shape[-2:]), force_not_quantize=True, shape=(1, *h.shape[-2:], h.shape[1])).sample
        res[size] = {"input": list(x.shape), "pre_quant": list(h.shape), "codes": int(idx.numel()),
                     "unique_codes": int(torch.unique(idx).numel()), "codebook": model.config.num_vq_embeddings,
                     "psnr_db": psnr(rec.cpu(), x.cpu()), "dtype": str(dt),
                     "time_encode": tokens.time_gpu(lambda: model.quantize(model.encode(x).latents))}
    save("vq", {"model": mid + " (vqvae)", "sizes": res})


@torch.no_grad()
def whisper():
    from transformers import WhisperFeatureExtractor, WhisperModel
    mid = "openai/whisper-tiny"
    fe = WhisperFeatureExtractor.from_pretrained(mid)
    enc = WhisperModel.from_pretrained(mid, dtype=DT, device_map=DEV).eval().encoder
    res = {}
    for sec in (1, 10, 30):
        a = tokens.tones(sec, 16000)
        feats = fe(a, sampling_rate=16000, return_tensors="pt").input_features
        out = enc(feats.to(DEV, DT)).last_hidden_state
        res[sec] = {"samples": len(a), "log_mel": list(feats.shape), "encoder": list(out.shape)}
    feats = fe(tokens.tones(30, 16000), sampling_rate=16000, return_tensors="pt").input_features.to(DEV, DT)
    save("whisper", {"model": mid, "clips": res, "time_encoder_30s": tokens.time_gpu(lambda: enc(feats))})


@torch.no_grad()
def mimi():
    from transformers import MimiModel, AutoFeatureExtractor
    mid = "kyutai/mimi"
    fe = AutoFeatureExtractor.from_pretrained(mid)
    model = MimiModel.from_pretrained(mid, dtype=torch.float32, device_map=DEV).eval()  # 96M params, small
    a = tokens.tones(10, 24000)
    iv = fe(raw_audio=a, sampling_rate=24000, return_tensors="pt")["input_values"].to(DEV)
    res = {"samples": len(a), "input": list(iv.shape)}
    for q in (8, 32):
        codes = model.encode(iv, num_quantizers=q).audio_codes
        audio = model.decode(codes).audio_values[..., : iv.shape[-1]]
        err = (audio - iv).float()
        snr = 10 * math.log10(iv.float().pow(2).mean().item() / err.pow(2).mean().item())
        res[q] = {"codes": list(codes.shape), "max_code": int(codes.max()), "snr_db": snr}
    res["time_encode_10s_q8"] = tokens.time_gpu(lambda: model.encode(iv, num_quantizers=8))
    res["codebook_size"] = model.config.codebook_size
    res["frame_rate"] = model.config.frame_rate
    save("mimi", {"model": mid, **res})


def procs():
    """Tokenizer-side counts only (no weights): image and video tokens from two VLM processors."""
    from transformers import AutoProcessor
    res = {}
    q = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    pad_id = q.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vpad_id = q.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    res["qwen2vl_image"] = {}
    for size in (224, 512, 1024):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        text = q.apply_chat_template(msgs, add_generation_prompt=True)
        enc = q(text=[text], images=[tokens.scene(size, size)], return_tensors="pt")
        res["qwen2vl_image"][size] = {"grid_thw": enc["image_grid_thw"].tolist()[0],
                                      "image_pad_tokens": int((enc["input_ids"] == pad_id).sum())}
    # 4 s of video at 448 x 448, already sampled at 2 frames per second (8 frames)
    frames = [np.asarray(f) for f in tokens.video(4.0, 2, 448, 448)]
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe."}]}]
    text = q.apply_chat_template(msgs, add_generation_prompt=True)
    enc = q(text=[text], videos=[np.stack(frames)], return_tensors="pt", do_sample_frames=False)
    res["qwen2vl_video_4s_2fps_448"] = {"grid_thw": enc["video_grid_thw"].tolist()[0],
                                        "video_pad_tokens": int((enc["input_ids"] == vpad_id).sum())}
    s = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    img_id = s.tokenizer.convert_tokens_to_ids("<image>")
    res["smolvlm_image"] = {}
    for size in (224, 512, 1024):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        text = s.apply_chat_template(msgs, add_generation_prompt=True)
        enc = s(text=text, images=[tokens.scene(size, size)], return_tensors="pt")
        res["smolvlm_image"][size] = {"pixel_values": list(enc["pixel_values"].shape),
                                      "image_tokens": int((enc["input_ids"] == img_id).sum()),
                                      "total_tokens": int(enc["input_ids"].shape[1])}
    save("procs", res)


@torch.no_grad()
def vivit():
    """ViViT-B/16x2 tubelet embedding, built from transformers' default VivitConfig (the
    google/vivit-b-16x2-kinetics400 architecture) with random weights: only shapes are measured."""
    from transformers import VivitConfig, VivitModel
    cfg = VivitConfig()
    m = VivitModel(cfg).to(DEV, DT).eval()
    clip = torch.randn(1, cfg.num_frames, 3, cfg.image_size, cfg.image_size, device=DEV, dtype=DT)
    emb = m.embeddings(clip)
    save("vivit", {"config": {"num_frames": cfg.num_frames, "image_size": cfg.image_size, "tubelet": cfg.tubelet_size,
                              "hidden": cfg.hidden_size}, "embeddings": list(emb.shape)})


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t0 = time.time()
        globals()[s]()
        print(f"{s} done in {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()
