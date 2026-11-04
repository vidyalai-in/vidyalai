from kokoro import KPipeline

g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)  # text to phonemes only
for text in ["Dr. Lee paid $45.", "Please record the record.", "Gen. Ortiz met Lt. Brown."]:
    for result in g2p(text):
        print(result.phonemes)
