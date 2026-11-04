"""Exp 2: WER vs SNR with white and babble noise on the first 200 utterances of the test-clean subset.
Usage: exp2_noise.py <model>. Babble: 6 talkers summed from 60 test-other utterances outside its subset.
The same seeded noise is used for every model. Writes runs/exp2_<model>.jsonl."""
import sys, json, time
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab")
import numpy as np, torch
import asr
name = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sub = json.load(open("runs/subsets.json"))
utt = {u["id"]: u for s in ("test-clean", "test-other") for u in asr.librispeech(s)}
ids = sub["test-clean"][:N]
speech = [asr.load_flac(utt[i]["path"]) for i in ids]
srcs = [asr.load_flac(utt[i]["path"]) for i in sub["babble_sources"]]
proc, model = asr.load_whisper(name)
asr.transcribe(proc, model, speech[:16], batch_size=16)  # warm-up
out = open(f"runs/exp2_{name}.jsonl", "w")
conds = [("clean", None)] + [(n, s) for n in ("white", "babble") for s in (20, 10, 5, 0)]
for noise, snr in conds:
    if noise == "clean":
        x = speech
    else:
        x = []
        for k, sp in enumerate(speech):
            rng = np.random.default_rng(10_000 * (noise == "babble") + k)
            nz = asr.white_noise(len(sp), rng) if noise == "white" else asr.babble(len(sp), srcs, rng, talkers=6)
            x.append(asr.mix_at_snr(sp, nz, snr))
    t0 = time.perf_counter()
    hyps, info = asr.transcribe(proc, model, x, batch_size=16)
    for i, h in zip(ids, hyps):
        out.write(json.dumps({"id": i, "noise": noise, "snr": snr, "ref": utt[i]["ref"], "hyp": h}) + "\n")
    out.flush()
    c = asr.word_errors([asr.normalize(utt[i]["ref"], "english") for i in ids], [asr.normalize(h, "english") for h in hyps])
    print(name, noise, snr, "WER", round(asr.wer(c), 4), round(time.perf_counter() - t0, 1), "s", flush=True)
