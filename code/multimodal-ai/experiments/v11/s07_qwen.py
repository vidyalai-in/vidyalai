"""Exp 2, optional local LLM stage: Qwen2.5-1.5B-Instruct (Apache-2.0) in bf16 on the GPU, greedy,
streamed with TextIteratorStreamer. Same system prompt and turns as the Claude run."""
import sys, json, time, re, threading
import sys
sys.path.insert(0, '/home/amit/.cache/vidyalai-scratch/mm/lab')
import voice as V
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
D = "/home/amit/.cache/vidyalai-scratch/v11"
NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You are a voice assistant. Your reply is read aloud, so answer in plain spoken English: "
          "one to three short sentences, no lists, no markdown.")
END = re.compile(r"[.!?](?=\s)")
turns = [l.strip() for l in open(f"{D}/data/user_turns.txt") if l.strip()]
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, device_map="cuda").eval()
out = open(f"{D}/runs/qwen_stream.jsonl", "w")

def run(t):
    ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    st = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    t0 = time.perf_counter()
    th = threading.Thread(target=model.generate, kwargs=dict(**ids, streamer=st, max_new_tokens=150, do_sample=False))
    th.start()
    text, tf, ts = "", None, None
    for d in st:
        now = time.perf_counter() - t0
        if tf is None and d.strip():
            tf = now
        text += d
        if ts is None and END.search(text):
            ts = now
    th.join(); te = time.perf_counter() - t0
    m = END.search(text)
    return dict(ttft=tf, t_first_sentence=ts or te, t_total=te, first_chars=len((text[: m.end()] if m else text).strip()),
                full_chars=len(text.strip()), n_tokens=len(tok(text).input_ids), text=text)

for _ in range(2):
    run(turns[0])
with V.GpuSampler() as g:
    for rep in range(1):
        for i, t in enumerate(turns):
            r = run(t); r.update(turn=i, rep=rep)
            out.write(json.dumps(r) + "\n"); out.flush()
json.dump({"gpu": g.summary(), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}, open(f"{D}/runs/qwen_meta.json", "w"))
print("peak MB", torch.cuda.max_memory_allocated() / 2**20, g.summary())
