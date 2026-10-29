"""Summaries of the causal-tracing runs (CPU only). Usage: python trace_summary.py gpt2 [gpt2-medium qwen]
Writes post/trace_<model>_summary.json."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/interp")
from lab import patching as P

ROOT = "/home/amit/.cache/vidyalai-scratch/p05"
r3 = lambda x: np.round(np.asarray(x, dtype=float), 4).tolist()
ROLES = ["first subject", "middle subject", "last subject", "first after", "further", "last"]

def boot_frac(x, clean, corr, n_boot=1000, seed=0):
    """normalized share over prompts (ratio of means) with nan-aware means; x [n, ...]."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    f = lambda ix: (np.nanmean(x[ix], 0) - corr[ix].mean()) / (clean[ix].mean() - corr[ix].mean())
    est = f(np.arange(n))
    bs = np.stack([f(rng.integers(0, n, n)) for _ in range(n_boot)])
    return est, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)

def boot_aie(x, corr, n_boot=1000, seed=0):
    """Average indirect effect: mean over prompts of (restored - corrupted)."""
    d = x - corr.reshape((-1,) + (1,) * (x.ndim - 1))
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    est = np.nanmean(d, 0)
    bs = np.stack([np.nanmean(d[rng.integers(0, n, n)], 0) for _ in range(n_boot)])
    return est, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)

for which in sys.argv[1:]:
    a = np.load(f"{ROOT}/runs/trace_{which}.npz")
    meta = json.load(open(f"{ROOT}/post/trace_{which}.json"))
    ids = a["gn_ids"]
    clean = a["clean_m"][:, ids].T              # [n, 3]: P, LD vs others, LD vs self(0)
    corr = a["gn_corrupt"]                      # [n, 3]
    out = {"model": which, "n_known": len(ids), "n_prompts": meta["n_prompts"], "window": meta["window"],
           "noise_samples": meta["noise_samples"], "sigma": meta["sigma_subject_embed"],
           "clean_p": P.bootstrap_mean(clean[:, 0]), "corrupt_p": P.bootstrap_mean(corr[:, 0]),
           "clean_ld": P.bootstrap_mean(clean[:, 1]), "corrupt_ld": P.bootstrap_mean(corr[:, 1]),
           "mlp0_split": meta["mlp0_split"], "mlp0_split_n": meta["mlp0_split_n"]}
    for kind in ("hidden", "mlp", "attn"):
        x = a[f"gn_{kind}_roles"]               # [n, L, 3, 6]
        e, lo, hi = boot_aie(x[:, :, 0], corr[:, 0])
        out[f"gn_{kind}_aie_p"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}         # [L, 6]
        for mi, mn in ((0, "p"), (1, "ld")):
            e, lo, hi = boot_frac(x[:, :, mi], clean[:, mi], corr[:, mi])
            out[f"gn_{kind}_frac_{mn}"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}
    for kind in ("mlp1_last_subj", "severed_mlp", "severed_attn", "hidden_last_subj"):
        x = a[f"gn_{kind}"]                     # [n, L, 3]
        e, lo, hi = boot_aie(x[:, :, 0], corr[:, 0])
        out[f"gn_{kind}_aie_p"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}
        for mi, mn in ((0, "p"), (1, "ld")):
            e, lo, hi = boot_frac(x[:, :, mi], clean[:, mi], corr[:, mi])
            out[f"gn_{kind}_frac_{mn}"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}
    # STR
    sc, sr = a["str_clean"], a["str_corrupt"]   # [m, 3]: P(city_i), LD vs others, LD vs partner city
    out["n_str_pairs"] = len(sc)
    out["str_clean"] = {k: P.bootstrap_mean(sc[:, i]) for i, k in enumerate(("p", "ld_others", "ld_pair"))}
    out["str_corrupt"] = {k: P.bootstrap_mean(sr[:, i]) for i, k in enumerate(("p", "ld_others", "ld_pair"))}
    for kind in ("hidden", "mlp", "attn"):
        x = a[f"str_{kind}_roles"]
        for mi, mn in ((0, "p"), (1, "ld"), (2, "ldpair")):
            e, lo, hi = boot_frac(x[:, :, mi], sc[:, mi], sr[:, mi])
            out[f"str_{kind}_frac_{mn}"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}
    for kind in ("mlp1_last_subj", "hidden_last_subj"):
        x = a[f"str_{kind}"]
        for mi, mn in ((0, "p"), (1, "ld"), (2, "ldpair")):
            e, lo, hi = boot_frac(x[:, :, mi], sc[:, mi], sr[:, mi])
            out[f"str_{kind}_frac_{mn}"] = {"est": r3(e), "lo": r3(lo), "hi": r3(hi)}
    # Zhang & Nanda's ratio: summed MLP (window) effect at the last subject token / at middle subject tokens
    if which == "gpt2":
        for corr_name in ("gn", "str"):
            for mn in ("p", "ld"):
                e = np.array(out[f"{corr_name}_mlp_frac_{mn}"]["est"])
                if not np.all(np.isnan(e[:, 1])):
                    out[f"ratio_last_mid_{corr_name}_{mn}"] = float(np.nansum(e[:, 2]) / np.nansum(e[:, 1]))
    json.dump(out, open(f"{ROOT}/post/trace_{which}_summary.json", "w"))
    print(f"== {which}: n_known {len(ids)}, pairs {len(sc)}, clean P {np.mean(clean[:,0]):.3f} -> GN {np.mean(corr[:,0]):.3f}; "
          f"LD {np.mean(clean[:,1]):.2f} -> {np.mean(corr[:,1]):.2f}; STR LDpair {np.mean(sc[:,2]):.2f} -> {np.mean(sr[:,2]):.2f}")
    for k in ("gn_hidden_aie_p", "gn_mlp_aie_p", "gn_attn_aie_p"):
        e = np.array(out[k]["est"]); print(k, "(cols = roles)"); print(np.round(e.T, 3))
    for k in ("gn_hidden_last_subj_frac_p", "gn_mlp1_last_subj_frac_p", "gn_severed_mlp_frac_p", "gn_severed_attn_frac_p"):
        print(k, np.round(out[k]["est"], 3))
    for c in ("gn", "str"):
        for mn in ("p", "ld"):
            e = np.array(out[f"{c}_mlp_frac_{mn}"]["est"]); print(f"{c} mlp window frac {mn} last subj", np.round(e[:, 2], 3), "last tok", np.round(e[:, 5], 3))
    print("str hidden frac ldpair (cols=roles)"); print(np.round(np.array(out["str_hidden_frac_ldpair"]["est"]).T, 2))
    print("str mlp1 last subj ldpair", np.round(out["str_mlp1_last_subj_frac_ldpair"]["est"], 3))
    print({k: v for k, v in out.items() if k.startswith("ratio")})
    print("mlp0 split", {k: np.round(v, 3).tolist() for k, v in out["mlp0_split"].items()})
