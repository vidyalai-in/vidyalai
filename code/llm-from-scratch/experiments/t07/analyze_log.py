# Part 7: summary numbers from the main run's log.jsonl (CPU only).
import json, sys, statistics as st, math
import numpy as np
P = sys.argv[1] if len(sys.argv) > 1 else "/home/amit/.cache/vidyalai-scratch/t07/runs/main88/log.jsonl"
recs = [json.loads(l) for l in open(P)]
train = [r for r in recs if "loss" in r]
val = [r for r in recs if "val_loss" in r]
ev = [r for r in recs if "event" in r]
print("events:", [(e["event"], e["step"], e.get("elapsed"), e.get("peak_mem_gib")) for e in ev])
print("train steps", len(train), "val points", len(val))
steps = np.array([r["step"] for r in train]); loss = np.array([r["loss"] for r in train])
gn = np.array([r["grad_norm"] for r in train]); dt = np.array([r["dt"] for r in train])
tps = np.array([r["tok_per_s"] for r in train]); el = np.array([r["elapsed"] for r in train])
lr = np.array([r["lr"] for r in train])
end = [e for e in ev if e["event"] == "end"][0]
T = 3360 * 524288
print(f"total tokens {T:,}; elapsed {end['elapsed']} s = {end['elapsed']/3600:.3f} h; overall {T/end['elapsed']:.0f} tok/s")
print(f"sum dt {dt.sum():.1f} s = {dt.sum()/3600:.3f} h; tok/s over step time {T/dt.sum():.0f}")
print(f"step0 dt {dt[0]}, elapsed at step0 {el[0]}; steps 1.. sum {dt[1:].sum():.1f}")
print(f"per-step tok/s (excluding step 0): median {np.median(tps[1:]):.0f} mean {tps[1:].mean():.0f} p5 {np.percentile(tps[1:],5):.0f} p95 {np.percentile(tps[1:],95):.0f} min {tps[1:].min()} (step {steps[1:][tps[1:].argmin()]}) max {tps[1:].max()} (step {steps[1:][tps[1:].argmax()]})")
print(f"dt median {np.median(dt[1:]):.3f} min {dt[1:].min():.3f} max {dt[1:].max():.3f}")
# gaps between steps: elapsed[i] - elapsed[i-1] - dt[i]
gaps = el[1:] - el[:-1] - dt[1:]
print(f"gaps total {gaps.sum():.1f} s; gap before step 0 (val0+start) {el[0]-dt[0]:.2f}")
# classify gaps: after a step s (step index s completed -> step s+1 counter). After step s (0-indexed) completes, counter=s+1; ckpt if (s+1)%100==0, snapshot if s+1 in set; val before step s+1 if (s+1)%50==0
snaps = {5,10,15,20,25,30,40,50,60,75,90,125,150,250}
cat = {"val only": [], "ckpt+snap (+val)": [], "snapshot only": [], "snapshot+val": [], "none": []}
for i in range(1, len(train)):
    n = int(steps[i])   # gap occurs before step n starts: after counter reached n
    g = gaps[i-1]
    isck = n % 100 == 0; isval = n % 50 == 0; issn = n in snaps
    if isck: cat["ckpt+snap (+val)"].append(g)
    elif issn and isval: cat["snapshot+val"].append(g)
    elif issn: cat["snapshot only"].append(g)
    elif isval: cat["val only"].append(g)
    else: cat["none"].append(g)
for k, v in cat.items():
    if v: print(f"  {k:20s} n={len(v):4d} total {sum(v):8.1f} s  median {st.median(v):.2f}  max {max(v):.2f}")
final_gap = end["elapsed"] - el[-1]
print(f"after last step (final ckpt? no: ckpt happens before elapsed logged?) end-last elapsed {final_gap:.2f}")
# val curve
V = [(r["step"], r["val_loss"]) for r in val]
print("val:", [(s, round(v, 4)) for s, v in V if s in (0,50,100,150,200,250,300,400,500,750,1000,1500,2000,2500,3000,3360)])
# unigram crossing
U = 7.59
for name, arr in (("train", list(zip(steps, loss))),):
    first = next(s for s, l in arr if l < U); print(f"train first below {U}: step {first}, tokens {(first+1)*524288:,}")
firstv = next((s, v) for s, v in V if v < U); print("val first below unigram:", firstv)
# smoothed train loss (EMA and centered window)
def window(x, w):
    out = np.convolve(x, np.ones(w)/w, mode="valid"); return out
# train-vs-val gap: val at step s measured BEFORE step s's update; compare with mean training loss of steps s-25..s+24
print("gap: val minus mean train loss of the 50 steps ending at s-1 / centred 10 steps")
for s, v in V:
    if s in (100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3360):
        m = loss[(steps >= s-10) & (steps < s)].mean(); m50 = loss[(steps >= s-50) & (steps < s)].mean()
        print(f"  step {s}: val {v:.4f}  train last10 {m:.4f} (gap {v-m:+.4f})  last50 {m50:.4f} (gap {v-m50:+.4f})")
gaps_v = []
for s, v in V:
    if s >= 200:
        m = loss[(steps >= s-5) & (steps < s+5)].mean(); gaps_v.append(v - m)
print(f"val - centred-10 train, steps>=200: mean {np.mean(gaps_v):+.4f} sd {np.std(gaps_v):.4f} min {min(gaps_v):+.4f} max {max(gaps_v):+.4f}")
# loss milestones
for thr in (7, 6, 5, 4.5, 4, 3.75, 3.5):
    i = next((k for k, (s, v) in enumerate(V) if v < thr), None)
    if i is not None: print(f"val first below {thr}: step {V[i][0]}, tokens {V[i][0]*524288/1e6:.0f}M")
# fraction of total loss drop by token fraction
v0, vend = V[0][1], V[-1][1]
for s, v in V:
    if s in (25*2, 100, 150, 250, 336, 500, 1000, 1680, 2240, 3360):
        print(f"  step {s} ({s/3360:.1%} of tokens): val {v:.4f}, {(v0-v)/(v0-vend):.1%} of total drop")
# drops per third
for a, b in ((0, 1120), (1120, 2240), (2240, 3360)):
    va = dict(V)[a] if a in dict(V) else None
print({s: round(v,4) for s, v in V if s in (1100,1150,1100,2200,2250,3350,3360)})
# per-500-step drops
VD = dict(V)
for a in range(0, 3360, 500):
    b = min(a + 500, 3350)
    print(f"  val {a}->{b}: {VD[a]:.4f} -> {VD[b]:.4f}  drop {VD[a]-VD[b]:.4f}")
# grad norm spikes
post = [(int(s), g, l) for s, g, l in zip(steps, gn, loss) if s >= 150 and g > 1.0]
print(f"steps >=150 with gn>1: {len(post)}:", [(s, round(g, 3)) for s, g, _ in post])
pre = [(int(s), round(g,2)) for s, g in zip(steps, gn) if s < 150 and g > 1.0]
print(f"steps <150 with gn>1: {len(pre)}; max {max(gn[:150]):.2f} at {int(steps[gn[:150].argmax()])}")
print("clipped total:", int((gn > 1).sum()))
print("median gn by range:", [(a, round(float(np.median(gn[(steps>=a)&(steps<a+100)])),3)) for a in (0,100,150,200,300,500,1000,2000,3000,3260)])
print(f"gn last 100 median {np.median(gn[-100:]):.3f}; min gn overall {gn.min():.3f} at {int(steps[gn.argmin()])}")
# loss around the two biggest spikes
for s0 in (207, 494):
    sl = [(int(s), round(float(l), 4), round(float(g), 3)) for s, l, g in zip(steps, loss, gn) if s0-4 <= s <= s0+4]
    print("around", s0, sl)
# batch-to-batch noise: std of (loss - centred 21-step mean) in late training
w = 21
sm = np.convolve(loss, np.ones(w)/w, mode="same")
res = (loss - sm)[w:-w]
print(f"residual sd (21-step centered) overall {res.std():.4f}; steps 1000-3300 {(loss-sm)[1000:3300].std():.4f}; max |res| {np.abs(res).max():.4f} at step {int(steps[w:-w][np.abs(res).argmax()])}")
r_late = (loss - sm)[300:3340]
idx = np.argsort(-r_late)[:8]
print("largest upward residuals after step 300:", [(int(steps[300+i]), round(float(r_late[i]), 3), round(float(gn[300+i]), 3)) for i in idx])
print(f"lr at 150 {lr[150]:.3e} at 1680 {lr[1680]:.3e} at 2240 {lr[2240]:.3e} last {lr[-1]:.3e}")
