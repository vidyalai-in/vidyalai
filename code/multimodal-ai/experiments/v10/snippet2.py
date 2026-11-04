from kokoro import KPipeline

pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
for r in pipe("Please record the record for the archive.", voice="af_heart"):
    print(len(r.phonemes), "phonemes ->", len(r.pred_dur), "durations, first six:", r.pred_dur[:6].tolist())
    print(len(r.audio), "samples =", len(r.audio) / 24000, "seconds")
