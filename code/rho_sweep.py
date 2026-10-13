"""rho_sweep.py — T13 (P1b, POST-HOC after CP-B1): low-rho operating regime, content-aware vs iso-energy periodic.

Question: is there a rho band where a content-aware gate beats the periodic schedule of the SAME per-clip open rate on
mean latency or on event miss? (CP-B1 §3 found periodic dominant at rho 0.77-0.82.)

  corpora   VIRAT (8 clips, real ASMAG scores), C121_all (121), C121_eval (12 = A1 evaluable at W = 1 s); no SYN
  families  G2_pure       S_t > tau (no certificate)
            G8_Pr{5,10,20} S_t > tau OR periodic refresh P_r in {L, 2L, 4L} (certificate L, 2L, 4L)
            tau is inverted PER CORPUS so that the mean per-clip open rate hits rho* in RHOS (bisection);
            rho* < 1/P_r is infeasible for G8 and recorded as such
  control   periodic at the content-aware gate's own per-clip rate (Bresenham, averaged over N_PHASE phases)
  metrics   mean latency (frames from onset to first open, uncapped), miss of short events (D < L), miss of long
            events (D >= L; no open inside the event), MI(E_W; A_W) A1 oracle W = 1 s (plug-in, permutation-corrected)
  win       diff = periodic - content-aware, pooled over events; 95 % bootstrap CI over videos (N_BOOT); a win needs
            the CI lower bound > 0

Outputs: results/P1b_rho_sweep.csv, results/P1b_win_regions.csv, results/P1b_fig_latency_vs_rho.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import psutil

import gates as G
import mi_estimators as M
from data_sparse import load_corpus
from schedules import load_config, results_dir

CFG = load_config()
L = CFG["certificate_L"]
OUT = results_dir(CFG)
RHOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
FAMILIES = [("G2_pure", None), ("G8_Pr5", L), ("G8_Pr10", 2 * L), ("G8_Pr20", 4 * L)]
N_PHASE = 8
N_SHUFFLE = 200
N_BOOT = 2000
SEED = 4213
MIN_CONTENT_SHARE = 0.05   # demand rate below 5 % of rho -> gate is effectively periodic, never counted as a win


def corpora():
    c121 = load_corpus("C121")
    ev = []
    for c in c121:
        w = int(round(c.fps))
        m = c.n // w
        a1 = c.e[: m * w].reshape(m, w).max(1)
        ev.append(M.h2(float(a1.mean())) >= 0.5 and m >= 30)
    return {"VIRAT": load_corpus("VIRAT"), "C121_all": c121, "C121_eval": [c for c, k in zip(c121, ev) if k]}


def phases(clips, P_r, rng):
    return [int(rng.integers(P_r)) for _ in clips] if P_r else [0] * len(clips)


def gate(c, tau, P_r, ph):
    d = G.demand_of(c.score, c.valid, tau)
    return d if P_r is None else np.maximum(d, G.periodic(c.n, P_r, ph))


def mean_rho(clips, tau, P_r, phs):
    return float(np.mean([gate(c, tau, P_r, p).mean() for c, p in zip(clips, phs)]))


def invert_tau(clips, target, P_r, phs):
    """Bisection on tau in [min score, max score + eps]; rho(tau) is non-increasing."""
    s = np.concatenate([c.score[c.valid == 1] for c in clips])
    lo, hi = float(s.min()) - 1e-9, float(s.max()) + 1e-9
    if mean_rho(clips, hi, P_r, phs) > target + 0.005:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mean_rho(clips, mid, P_r, phs) > target:
            lo = mid
        else:
            hi = mid
    return hi


def event_stats(a, e):
    """Per event (onset, D, latency, miss) — latency uncapped, miss = no open inside the event."""
    on = np.flatnonzero(np.diff(np.r_[0, e]) == 1)
    off = np.flatnonzero(np.diff(np.r_[e, 0]) == -1)
    idx = np.flatnonzero(a)
    pos = np.searchsorted(idx, on)
    nxt = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], len(a)) if len(idx) else np.full(len(on), len(a))
    D = off - on + 1
    return D, (nxt - on).astype(float), (nxt > off).astype(float)


def periodic_rate(n, rho, phase):
    t = np.arange(n)
    return (np.floor((t + 1) * rho + phase) - np.floor(t * rho + phase)).astype(np.int8)


def clip_metrics(a, e):
    D, lat, miss = event_stats(a, e)
    return dict(D=D, lat=lat, miss=miss)


def mi_a1(a, c, rng):
    w = int(round(c.fps))
    m = c.n // w
    if m < 4:
        return np.nan
    e = c.e[: m * w].reshape(m, w).max(1).astype(np.int8)
    rate = a[: m * w].reshape(m, w).mean(1)
    return M.plugin(e, rate, rng, N_SHUFFLE)[2]


def pooled(per_clip, key, sel):
    num = sum(float(m[key][sel(m)].sum()) for m in per_clip)
    den = sum(int(sel(m).sum()) for m in per_clip)
    return num / den if den else np.nan, num, den


SELS = {"lat": ("lat", lambda m: np.ones(len(m["D"]), bool)),
        "miss_short": ("miss", lambda m: m["D"] < L),
        "miss_long": ("miss", lambda m: m["D"] >= L)}


def boot_diff(ca, pe, metric, rng):
    """diff = periodic - content-aware (positive = content-aware better), bootstrap over videos."""
    key, sel = SELS[metric]
    n = len(ca)
    stats = np.array([[float(ca[i][key][sel(ca[i])].sum()), float(pe[i][key][sel(pe[i])].sum()),
                       int(sel(ca[i]).sum())] for i in range(n)])
    if stats[:, 2].sum() == 0:
        return np.nan, np.nan, np.nan, 0
    point = (stats[:, 1].sum() - stats[:, 0].sum()) / stats[:, 2].sum()
    idx = rng.integers(n, size=(N_BOOT, n))
    num = stats[idx, 1].sum(1) - stats[idx, 0].sum(1)
    den = stats[idx, 2].sum(1)
    b = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    return point, float(lo), float(hi), int(stats[:, 2].sum())


def boot_mean(x, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def run_corpus(name, clips):
    rows = []
    for fam, P_r in FAMILIES:
        for rho_t in RHOS:
            rng = np.random.default_rng([SEED, len(name), RHOS.index(rho_t), P_r or 0])
            phs = phases(clips, P_r, rng)
            tau = invert_tau(clips, rho_t, P_r, phs)
            base = dict(corpus=name, family=fam, P_r=P_r or np.nan, rho_target=rho_t, n_videos=len(clips))
            if tau is None:
                rows.append(dict(base, feasible=0))
                continue
            ca, pe, mi_ca, mi_pe, rho_c, dem_c = [], [], [], [], [], []
            for c, ph in zip(clips, phs):
                a = gate(c, tau, P_r, ph)
                r = float(a.mean())
                rho_c.append(r)
                dem_c.append(float(G.demand_of(c.score, c.valid, tau).mean()))
                ca.append(clip_metrics(a, c.e))
                # periodic at the same per-clip rate, latency/miss averaged over N_PHASE phases
                acc = None
                for k in range(N_PHASE):
                    m = clip_metrics(periodic_rate(c.n, max(r, 1e-6), (k + 0.5) / N_PHASE), c.e)
                    acc = m if acc is None else dict(D=m["D"], lat=acc["lat"] + m["lat"], miss=acc["miss"] + m["miss"])
                pe.append(dict(D=acc["D"], lat=acc["lat"] / N_PHASE, miss=acc["miss"] / N_PHASE))
                mi_ca.append(mi_a1(a, c, rng))
                mi_pe.append(mi_a1(periodic_rate(c.n, max(r, 1e-6), rng.random()), c, rng))
            r = dict(base, feasible=1, tau=tau, rho=float(np.mean(rho_c)), demand_rate=float(np.mean(dem_c)))
            # degenerate = (almost) no content-driven opens: the gate IS a periodic schedule
            r["degenerate"] = int(r["demand_rate"] < MIN_CONTENT_SHARE * r["rho"])
            for met in SELS:
                key, sel = SELS[met]
                r[f"{met}_ca"] = pooled(ca, key, sel)[0]
                r[f"{met}_per"] = pooled(pe, key, sel)[0]
                d, lo, hi, ne = boot_diff(ca, pe, met, rng)
                r.update({f"{met}_diff": d, f"{met}_diff_lo": lo, f"{met}_diff_hi": hi, f"{met}_n_events": ne,
                          f"{met}_win": int(lo > 0 and not r["degenerate"]) if not np.isnan(lo) else 0,
                          f"{met}_loss": int(hi < 0) if not np.isnan(hi) else 0})
            r["MI_ca"], r["MI_ca_lo"], r["MI_ca_hi"] = boot_mean(mi_ca, rng)
            r["MI_per"], r["MI_per_lo"], r["MI_per_hi"] = boot_mean(mi_pe, rng)
            rows.append(r)
            print(f"{name:9s} {fam:8s} rho*={rho_t:.2f} rho={r['rho']:.3f} tau={tau:.3f} dem={r['demand_rate']:.3f} "
                  f"lat {r['lat_ca']:.1f}/{r['lat_per']:.1f} mS {r['miss_short_ca']:.2f}/{r['miss_short_per']:.2f} "
                  f"mL {r['miss_long_ca']:.2f}/{r['miss_long_per']:.2f} MI {r['MI_ca']:.3f}", flush=True)
    return rows


def win_regions(df):
    out = []
    for (corpus, fam), g in df[df.feasible == 1].groupby(["corpus", "family"], sort=False):
        for met in SELS:
            w = g[g[f"{met}_win"] == 1]
            out.append(dict(corpus=corpus, family=fam, metric=met, n_rho=len(g),
                            n_degenerate=int(g.degenerate.sum()), n_win=len(w),
                            n_loss=int(g[f"{met}_loss"].sum()),
                            rho_win=";".join(f"{x:.2f}" for x in w.rho_target),
                            best_diff=float(g[f"{met}_diff"].max())))
    return pd.DataFrame(out)


def fig(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cors = list(dict.fromkeys(df.corpus))
    mets = [("lat", "mean latency (frames)"), ("miss_short", f"miss, D < {L}"), ("miss_long", f"miss, D >= {L}"),
            ("MI", "MI A1, W = 1 s (bit)")]
    fig, ax = plt.subplots(len(cors), len(mets), figsize=(15, 3.3 * len(cors)), squeeze=False)
    cols = dict(zip([f for f, _ in FAMILIES], ["C0", "C1", "C2", "C3"]))
    for i, cname in enumerate(cors):
        g = df[(df.corpus == cname) & (df.feasible == 1)]
        for j, (m, lab) in enumerate(mets):
            a = ax[i, j]
            for fam, gg in g.groupby("family", sort=False):
                gg = gg.sort_values("rho")
                a.plot(gg.rho, gg[f"{m}_ca"], "o-", color=cols[fam], label=f"{fam}")
                a.plot(gg.rho, gg[f"{m}_per"], "x--", color=cols[fam], alpha=0.6, label=f"periodic iso-ρ ({fam})")
            if g[f"{m}_ca"].isna().all():
                a.text(0.5, 0.5, f"no events with D < {L}", ha="center", va="center", transform=a.transAxes)
            a.set_xlabel("open rate ρ")
            a.set_title(f"{cname}: {lab}", fontsize=9)
            if m == "lat":
                a.set_yscale("log")
        if i == 0:
            ax[0, 0].legend(fontsize=6)
    fig.suptitle("P1b (post-hoc): content-aware (solid) vs periodic at the same per-clip ρ (dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "P1b_fig_latency_vs_rho.png", dpi=150)


def main():
    free = psutil.virtual_memory().available / 2**30
    if free < CFG["virat"]["min_free_ram_gb"]:
        raise SystemExit(f"free RAM {free:.1f} GB < {CFG['virat']['min_free_ram_gb']} GB")
    rows = []
    for name, clips in corpora().items():
        print(f"{name}: {len(clips)} clips, {sum(c.n for c in clips)} frames", flush=True)
        rows += run_corpus(name, clips)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "P1b_rho_sweep.csv", index=False)
    w = win_regions(df)
    w.to_csv(OUT / "P1b_win_regions.csv", index=False)
    fig(df)
    print(w.to_string())


if __name__ == "__main__":
    main()
