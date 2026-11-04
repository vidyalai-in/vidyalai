"""Probe: which attention does Qwen2-VL's vision tower use here, and what does it cost in time and memory?"""
import sys, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import serving as S, vlm
m = vlm.VLM("Qwen/Qwen2-VL-2B-Instruct")
M = m.model
print("impl:", M.config._attn_implementation, M.config.vision_config._attn_implementation,
      M.model.visual.config._attn_implementation)
for impl in (M.model.visual.config._attn_implementation, "sdpa"):
    M.model.visual.config._attn_implementation = impl
    for w, h in ((672, 448), (1344, 896)):
        x = m.inputs(S.report_page(0, w, h), "Q?")
        enc = {k: x[k] for k in ("pixel_values", "image_grid_thw")}
        with torch.inference_mode():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ts = S.cuda_times(lambda: M.model.get_image_features(**enc, return_dict=True), warmup=1, reps=3)
        print(impl, w, h, [round(t, 1) for t in ts], "peak MB", round(torch.cuda.max_memory_allocated() / 2**20))
print("free/total MB", [round(v / 2**20) for v in torch.cuda.mem_get_info()])
