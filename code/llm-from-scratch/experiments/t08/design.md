# t08 phase A notes (design log)
- Sizes: fromscratch/sizes.py (s11 s15 s22=SMALL s33 s47). train.py: +`from sizes import SIZES`, CONFIGS adds **SIZES. Backup notes/train.py.orig.
- Throughput (15 steps, 131,072 tok/step, compiled+fused, mb16): s11 186,944; s15 130,368; s22 110,452; s33 73,104; s47 53,440 tok/s.
  mb32: s11 187,348 (3.56 GiB); s15 131,922 (4.19); s22 111,049 (4.95); s33 42,399 (6.30 GiB, spills -> slow). -> mb16 everywhere.
- Batch rule candidate R1: tokens/step ∝ N (524,288 x N/88.1M), nearest power of 2 => ~3,360 steps like main88:
  s11 65,536; s15 65,536; s22 131,072; s33 262,144; s47 262,144; main 524,288.
- Pilot set 1: s11 at 25% budget (54.69M tokens), warm-up 4.5% of steps.
- Pilot set 1 results (s11, 54.7M tokens, val 40 batches): 65k: 7.5e-4 5.276; 1.5e-3 5.053; 3e-3 4.962; 6e-3 5.467. 131k @6e-3 6.057.
  65k-batch throughput 178k tok/s (vs 187k at 131k: per-step overhead).
- Pilot set 2: s11 131k @3e-3 (batch check); s47 262k @1.5e-3 and @3e-3 at 5% budget (47.2M tokens, 180 steps, warm-up 8).
- Pilot set 2: s11 131k @3e-3: 6.008 (vs 4.962 at 65k: same tokens, half the steps; 131k run sat on the unigram
  plateau ~7.4-7.5 through step 84 = 11M tokens; 65k run was at 6.68 by step 84). s47 262k, 180 steps (47.2M tok):
  1.5e-3 6.119; 3e-3 6.510. s47 throughput 51.5k tok/s over 180 steps (1022 s elapsed incl. compile + 11 evals).
- Smoke test of run_sweep.sh (ROOT=smoke, s11+s15, --stop-at 40 --ckpt-every 10, kill -9 at step 15): resumed from
  step 10 on try 2, ended at 40; second launch skipped both. Smoke dir deleted. s11 65k 186k tok/s, s15 65k 132k tok/s (cool GPU).
- GPU time phase A: 4,595 s = 76.6 min (gpu_ledger.txt).

## Decisions (final)
- Sizes: s11 (240/5, 3/1 heads, hd 80, ff 640), s15 (288/6, 6/2, hd 48, ff 768), s22 = SMALL, s33 (480/7, 6/2, hd 80, ff 1280),
  s47 (576/8, 9/3, hd 64, ff 1536 = Part 1's 47M). GQA 3:1 everywhere; ff = 8/3 d exactly; d/L 48,48,64,69,72 (main 77).
- Budget N = total params (Part 1 convention); fit both total and non-embedding in phase B.
- Batch: tokens/step ~ 524,288 x N/88.1M rounded to power of 2 -> about 3,360 steps per size (2,513-4,501).
  Why: Part 5 + pilot: the unigram-plateau escape costs ~100 steps whatever the batch, so equal steps treats sizes alike;
  Kaplan B_crit falls with loss; accumulation makes small batches free in throughput (65k costs ~5% at s11).
- LR: peak = 0.6/d (1/width, muP-style for Adam). s11 2.5e-3 (pilot valley: 1.5e-3 5.053 / 3e-3 4.962 / 6e-3 5.467,
  parabola min ~2.4e-3); s47 1.04e-3 (pilot: 1.5e-3 beat 3e-3 by 0.39; low side untested, but s11 valley is flat below
  the optimum (+0.09 at half) and steep above (+0.5 at double), and full-length runs prefer lower rates than pilots).
  main88 6e-4 = 0.46/d, i.e. 0.77x this rule.
- Warm-up 4.5% of steps (150/3,360 as Part 7); cosine to 10%; wd 0.1; clip 1.0; seed 1337; val every ~2.5% (40 batches),
  ckpt + weights-only snapshot every ~5% (--keep-ckpts) -> final snapshot model_<steps>.pt per run.
- IsoFLOP at C(s15) = 6 x 14,749,344 x 294,977,536 = 2.61e16: iso_s11 (36.4 tok/p), s15 main (20), iso_s22 (9.0).
  iso_s33 (4.0 tok/p, 504 steps, ~0.5 h) cut to stay under ~12 h.
- Predicted (notes/plan.txt, sustained tok/s s11 178k, s15 125k, s22 105k, s33 70k, s47 51.5k): main 9.87 h of steps,
  +iso 1.14 h = 11.0 h; with compile/eval/ckpt overhead about 11.3 h.
- Launch: cd t08/scripts && nohup flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash run_sweep.sh > ../logs/sweep.nohup 2>&1 &

## Phase B (2026-10-28)
- Sweep 13:19:53 -> 01:20:01 (12 h 00 min), one try each, exit 0, no divergence (logs/sweep.out). post/analyze.out per run.
- Whole-shard val (post/full_val.jsonl): s11 4.1201, s15 3.9731, s22 3.7973, s33 3.6638, s47 3.5135, iso_s11 4.0037, iso_s22 4.0828; main88 3.3651 (t07).
- Seed repeat s11 seed 1338 (runs/s11_seed1338, 23 min, ckpt.pt only): subset 4.1453 (vs 4.1346), whole 4.1289 (vs 4.1201): diff 0.0088.
- Fits post/fit.out; isoflop MC post/iso_mc.out; throughput post/tput_clock.out. Median-vs-mean: planning used median step tok/s; ~2 s host stall every ~34 s (nvidia-smi log gaps too) on small runs.
- Charts: diagrams/t08-loss-compute.py, t08-scaling-fit.py, t08-isoflop.py -> public/diagrams.
- Snippets: snip/fit_snippet.py(.out), snip/val_snippet.py(.out).
