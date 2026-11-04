import sys, json, time; sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import speech_tts as st
from kokoro import KPipeline
pa = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
pb = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M', model=False)
rows = [json.loads(l) for l in open("texts/hard.jsonl")]
out = []
for r in rows:
    r["ps_us"] = " ".join(p for _, p in st.phonemize(pa, r["text"]))
    r["ps_gb"] = " ".join(p for _, p in st.phonemize(pb, r["text"]))
    out.append(r); print(r["id"], "|", r["text"], "\n   US:", r["ps_us"], "\n   GB:", r["ps_gb"])
json.dump(out, open("runs_g2p_hard.json", "w"), ensure_ascii=False, indent=1)
