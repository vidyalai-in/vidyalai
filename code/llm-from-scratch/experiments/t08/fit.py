"""fit.py: power-law fits to the sweep's final losses (whole validation shard), with uncertainty,
plus the IsoFLOP parabola and the published laws evaluated at our sizes. Writes post/fit.json."""
import json
import numpy as np

names = ["s11", "s15", "s22", "s33", "s47", "main88"]
N_tot = np.array([10_938_960, 14_749_344, 22_025_088, 32_939_040, 47_195_712, 88_096_512], float)
N_ne = np.array([3_074_640, 5_312_160, 9_442_176, 17_210_400, 28_321_344, 62_930_688], float)
fv = {json.loads(l)["run"]: json.loads(l) for l in open("post/full_val.jsonl")}
L = np.array([fv[n]["whole"] for n in names[:5]] + [3.3651])
out = {"L": dict(zip(names, L.round(4).tolist()))}

def power_law(N, L):
    """L = A * N^-a: a straight line in logs."""
    slope, logA = np.polyfit(np.log(N), np.log(L), 1)
    return -slope, np.exp(logA)

def saturating(N, L, grid=np.arange(0.02, 2.0, 0.0005)):
    """L = E + A * N^-alpha: for each alpha, E and A by linear least squares; keep the best alpha."""
    best = None
    for a in grid:
        X = np.stack([np.ones_like(N), N ** -a], 1)
        (E, A), *_ = np.linalg.lstsq(X, L, rcond=None)
        sse = ((X @ [E, A] - L) ** 2).sum()
        if A > 0 and (best is None or sse < best[0]):
            best = (sse, a, E, A)
    return best[1], best[2], best[3], np.sqrt(best[0] / len(N))

rng = np.random.default_rng(0)
for nlab, N in (("total", N_tot), ("non-embedding", N_ne)):
    for set_lab, k in (("sweep only (5)", 5), ("with 88M (6)", 6)):
        n, l = N[:k], L[:k]
        a_pl, A_pl = power_law(n, l)
        rms_pl = np.sqrt(((A_pl * n ** -a_pl - l) ** 2).mean())
        al, E, A, rms = saturating(n, l)
        loo_pl = [power_law(np.delete(n, i), np.delete(l, i))[0] for i in range(k)]
        loo_sat = [saturating(np.delete(n, i), np.delete(l, i))[:2] for i in range(k)]
        fit = E + A * n ** -al
        res = l - fit
        boot = []
        for _ in range(1000):                     # residual bootstrap of the 3-parameter fit
            lb = fit + rng.choice(res, k, replace=True) * np.sqrt(k / (k - 3))
            boot.append(saturating(n, lb, np.arange(0.02, 2.0, 0.002))[:2])
        boot = np.array(boot)
        r = {"power_law": {"a": a_pl, "A": A_pl, "rms": rms_pl, "loo_a": [min(loo_pl), max(loo_pl)]},
             "saturating": {"alpha": al, "E": E, "A": A, "rms": rms,
                            "loo_alpha": [min(x[0] for x in loo_sat), max(x[0] for x in loo_sat)],
                            "loo_E": [min(x[1] for x in loo_sat), max(x[1] for x in loo_sat)],
                            "boot_alpha_95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                            "boot_E_95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                            "residuals": res.round(4).tolist()}}
        if k == 5:                                 # extrapolate to the 88M model
            r["pred_88M"] = {"power_law": A_pl * N[5] ** -a_pl, "saturating": E + A * N[5] ** -al}
        out[f"{nlab} | {set_lab}"] = r
        print(f"{nlab:13} {set_lab:15} power law: a={a_pl:.4f} (LOO {min(loo_pl):.4f}-{max(loo_pl):.4f}) A={A_pl:.3f} rms={rms_pl:.4f} | "
              f"E+A/N^alpha: alpha={al:.3f} (LOO {r['saturating']['loo_alpha'][0]:.3f}-{r['saturating']['loo_alpha'][1]:.3f}, "
              f"boot95 {r['saturating']['boot_alpha_95'][0]:.3f}-{r['saturating']['boot_alpha_95'][1]:.3f}) "
              f"E={E:.3f} (LOO {r['saturating']['loo_E'][0]:.2f}-{r['saturating']['loo_E'][1]:.2f}) A={A:.4g} rms={rms:.4f}"
              + (f" | 88M predicted: pl {r['pred_88M']['power_law']:.4f}, sat {r['pred_88M']['saturating']:.4f} (actual 3.3651)" if k == 5 else ""))

# Published laws along D = 20 N, at our six sizes (their data and tokenizer: compare shapes, not levels)
laws = {"Chinchilla (Hoffmann et al., eq. 10)": (1.69, 406.4, 410.7, 0.34, 0.28),
        "Chinchilla, unrounded (Besiroglu et al. eq. 4)": (1.6934, 406.4, 410.7, 0.3392, 0.2849),
        "Epoch refit (Besiroglu et al., eq. 3)": (1.8172, 482.01, 2085.43, 0.3478, 0.3658)}
out["published"] = {}
for lab, (E, A, B, al, be) in laws.items():
    Lc = E + A * N_tot ** -al + B * (20 * N_tot) ** -be
    a_pl, _ = power_law(N_tot, Lc)
    red = Lc - E
    a_red = -np.polyfit(np.log(N_tot), np.log(red), 1)[0]   # exponent of the reducible part
    out["published"][lab] = {"L": Lc.round(3).tolist(), "pure_power_law_a": a_pl, "reducible_exponent": a_red}
    print(f"{lab:48} L at 11M..88M: {Lc[0]:.3f}..{Lc[-1]:.3f}  pure power-law slope {a_pl:.4f}  reducible-part exponent {a_red:.3f}")

# Kaplan: L(N, D) = [(Nc/N)^(aN/aD) + Dc/D]^aD with Table 2 values, N non-embedding, D = 20 x total N
aN, aD, Nc, Dc = 0.076, 0.103, 6.4e13, 1.8e13
Lk = ((Nc / N_ne) ** (aN / aD) + Dc / (20 * N_tot)) ** aD
Lk_inf = (8.8e13 / N_ne) ** 0.076
out["kaplan"] = {"L_ND": Lk.round(3).tolist(), "slope_ND_vs_Nne": power_law(N_ne, Lk)[0], "L_N_converged": Lk_inf.round(3).tolist()}
print(f"Kaplan L(N,D=20N) {Lk[0]:.3f}..{Lk[-1]:.3f}, slope vs non-emb N {power_law(N_ne, Lk)[0]:.4f}; L(N) eq 1.1 slope 0.076")

# IsoFLOP at C = 6 x 14,749,344 x 294,977,536
iso = {"iso_s11": (10_938_960, 397_737_984), "s15": (14_749_344, 294_977_536), "iso_s22": (22_025_088, 197_525_504)}
x = np.log([iso[k][0] for k in iso]); y = np.array([fv[k]["whole"] for k in iso])
c2, c1, c0 = np.polyfit(x, y, 2)
xo = -c1 / (2 * c2); No = np.exp(xo); C = 6 * 14_749_344 * 294_977_536; Do = C / (6 * No)
out["isoflop"] = {"C": C, "points": {k: {"N": iso[k][0], "D": iso[k][1], "tpp": iso[k][1] / iso[k][0], "L": fv[k]["whole"]} for k in iso},
                  "N_opt": No, "D_opt": Do, "tpp_opt": Do / No, "L_min": c0 + c1 * xo + c2 * xo ** 2, "curvature": c2}
print(f"IsoFLOP C={C:.3e}: " + ", ".join(f"{k} {iso[k][1]/iso[k][0]:.1f} tok/p L={fv[k]['whole']:.4f}" for k in iso)
      + f" -> parabola min N={No/1e6:.2f}M, D={Do/1e6:.0f}M, {Do/No:.1f} tok/p, L={out['isoflop']['L_min']:.4f}")
for lab, (E, A, B, al, be) in laws.items():
    G = (al * A / (be * B)) ** (1 / (al + be)); n_opt = G * (C / 6) ** (be / (al + be))
    out["isoflop"][f"N_opt {lab}"] = n_opt
    print(f"  {lab}: N_opt at this C = {n_opt/1e6:.1f}M, {C/6/n_opt/n_opt:.1f} tok/p")
json.dump(out, open("post/fit.json", "w"), indent=1, default=float)
