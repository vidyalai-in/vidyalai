"""Silero VAD on the 1,000 LibriSpeech subset clips (CPU): how many does it pass as speech?"""
import sys, json, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); import asr
from silero_vad import load_silero_vad, get_speech_timestamps
torch.set_num_threads(4)
vad = load_silero_vad(); sub = json.load(open("runs/subsets.json"))
utt = {u["id"]: u for s in ("test-clean", "test-other") for u in asr.librispeech(s)}
res = {}; t0 = time.time(); tot = 0
for sp in ("test-clean", "test-other"):
    miss = []
    for i in sub[sp]:
        a = asr.load_flac(utt[i]["path"]); tot += len(a) / 16000
        ts = get_speech_timestamps(torch.from_numpy(a), vad, sampling_rate=16000)
        if not ts: miss.append(i)
    res[sp] = {"n": len(sub[sp]), "no_speech_found": len(miss), "ids": miss}
res["cpu_s"] = time.time() - t0; res["audio_s"] = tot
json.dump(res, open("runs/vad_speech.json", "w"), indent=1); print(res)
