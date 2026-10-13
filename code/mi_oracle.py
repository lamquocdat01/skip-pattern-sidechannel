"""mi_oracle.py — P0 sanity: how much does an open schedule leak about scene events, per video?

For each video, gate family/parameter and window length W (non-overlapping windows):
  targets  E_pres  = 1[some in-event frame in window]   (attack goal A1)
           E_onset = 1[some event onset in window]       (coarse A2)
  observation O = N_W = number of opens in the window, binned into <= 8 equal-width bins of [0, W].
  N_W is a function of A_W, so I(E; N_W) <= I(E; A_W) (data processing): the reported value is a
  LOWER bound on the oracle leakage of the full pattern, up to estimator error.
Estimator: plug-in MI in bits; bias corrected by subtracting the mean of a within-video permutation
null (n_shuffle shuffles of E across windows); p = share of null >= observed.
Noisy rows: same pipeline with N_W recovered from a 1 Hz power trace or from encrypted-traffic timing.

Outputs (results/):
  P0_open_rate_by_video.csv  T04 table, 121 videos x open rate (deployed trace + threshold replay)
  P0_mi_oracle.csv           per video x family x param x observer x W x target
  P0_mi_summary.csv          aggregated over video sets (all / informative) with bootstrap CI
  P0_mi_oracle.png           MI vs open rate per family
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import observer as obs
from schedules import (event_runs, family_grid, load_config, load_videos, make_schedule,
                       processed_dir, results_dir)

N_BINS = 8


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.where(p > 0, p, 1.0)
    return -(p * np.log2(p)).sum(-1) * 1.0


def mi_batch(x: np.ndarray, y: np.ndarray, ny: int) -> np.ndarray:
    """Plug-in I(X;Y) in bits for binary X rows (S, n) against one Y vector (n,) with ny symbols."""
    S, n = x.shape
    idx = x.astype(np.int64) * ny + y[None, :] + (np.arange(S)[:, None] * 2 * ny)
    joint = np.bincount(idx.ravel(), minlength=S * 2 * ny).reshape(S, 2, ny) / n
    px, py = joint.sum(2), joint.sum(1)
    return entropy_bits(px) + entropy_bits(py) - entropy_bits(joint.reshape(S, -1))


def mi_with_null(e: np.ndarray, o: np.ndarray, rng, n_shuffle: int):
    n = len(e)
    if n < 4 or e.min() == e.max():
        return 0.0, 0.0, 0.0, 1.0
    obs_mi = float(mi_batch(e[None, :], o, N_BINS)[0])
    perm = np.argsort(rng.random((n_shuffle, n)), axis=1)
    null = mi_batch(e[perm], o, N_BINS)
    return obs_mi, float(null.mean()), max(0.0, obs_mi - float(null.mean())), float((null >= obs_mi - 1e-12).mean())


def windows(x: np.ndarray, w: int) -> np.ndarray:
    m = len(x) // w
    return x[: m * w].reshape(m, w)


def bin_counts(counts: np.ndarray, w: int) -> np.ndarray:
    return np.clip(np.floor(counts / (w + 1e-9) * N_BINS).astype(int), 0, N_BINS - 1)


def targets(v, w: int):
    pres = windows(v.e, w).max(1).astype(np.int8)
    on = np.zeros(v.n, np.int8)
    on[event_runs(v.e)[:, 0]] = 1
    onset = windows(on, w).max(1).astype(np.int8)
    return {"E_pres": pres, "E_onset": onset}


def binary_entropy(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def main() -> None:
    cfg = load_config()
    mcfg, ocfg = cfg["mi"], cfg["observer"]
    rng = np.random.default_rng(mcfg["seed"])
    out = results_dir(cfg)
    vids = load_videos(cfg)
    dg = cfg["deployed_gate"]

    # ---------------- T04: open rate per video
    act = pd.read_csv(processed_dir(cfg) / cfg["data"]["activation_by_video"]).set_index(["dataset", "video"])
    rows = []
    for v in vids:
        a = make_schedule(v, "content_aware_shield", dict(tau=dg["tau"], shield=True, T_refresh=dg["T_refresh"],
                                                          W_maxskip=dg["W_maxskip"], reuse_logic=dg["reuse_logic"]), rng)
        r = act.loc[(v.dataset, v.video)]
        rows.append(dict(dataset=v.dataset, video=v.video, n_frames=v.n,
                         event_frame_rate=round(float(v.e.mean()), 4), n_events=len(event_runs(v.e)),
                         open_rate_deployed=round(r.n_opens_deployed / v.n, 4),
                         open_rate_replay=round(float(a.mean()), 4)))
    orate = pd.DataFrame(rows)
    thr = mcfg["informative_open_rate_max"]
    orate["informative"] = (orate.open_rate_deployed < thr).astype(int)
    orate.to_csv(out / "P0_open_rate_by_video.csv", index=False)
    inform = set(orate.loc[orate.informative == 1, ["dataset", "video"]].astype(str).agg("/".join, axis=1))
    print(f"T04: {len(inform)}/{len(vids)} informative videos (deployed open rate < {thr})")

    # ---------------- MI per video
    noisy_cfgs = {("content_aware_shield", "tau=0.55"), ("content_aware_shield", "tau=0.78"),
                  ("content_aware_pure", "tau=0.55"), ("content_aware_pure", "tau=0.78"),
                  ("rate_limited", "K=5/W=25"), ("privacy_certified", "lam=0.2"), ("periodic", "P=5")}
    recs = []
    t0 = time.time()
    grid = list(family_grid(cfg))
    for gi, (fam, lab, p) in enumerate(grid):
        for v in vids:
            fps = cfg["fps"][v.dataset]
            a = make_schedule(v, fam, p, rng)
            rho = float(a.mean())
            views = {"oracle": a}
            if (fam, lab) in noisy_cfgs:
                pw = obs.power_trace(a, fps, ocfg, 1.0, 0.05, rng)                # 1 Hz plug, sigma 50 mW
                cnt = np.clip(np.round(obs.power_to_open_count(pw, fps, ocfg, 1.0)), 0, None)
                views["power_1Hz_s0.05"] = np.repeat(cnt, int(round(fps)))       # spread over frames; summed below
                views["power_1Hz_s0.05"] = views["power_1Hz_s0.05"] / int(round(fps))
                ts, _ = obs.traffic_trace(a, fps, ocfg, rng)
                views["traffic_timing"] = ts
            for W_s in mcfg["windows_s"]:
                w = int(round(W_s * fps))
                tg = targets(v, w)
                m = v.n // w
                for oname, ov in views.items():
                    if oname == "traffic_timing":
                        counts = obs.traffic_to_open_count(ov, v.n, fps, w)[:m]
                    else:
                        counts = windows(np.pad(ov, (0, max(0, m * w - len(ov)))), w).sum(1)[:m]
                    ob = bin_counts(counts, w)
                    for tname, e in tg.items():
                        if len(e) != len(ob):
                            k = min(len(e), len(ob)); e, ob_ = e[:k], ob[:k]
                        else:
                            ob_ = ob
                        if len(e) == 0:
                            continue
                        mi, null, corr, pval = mi_with_null(e, ob_, rng, mcfg["n_shuffle"])
                        recs.append(dict(dataset=v.dataset, video=v.video, family=fam, param=lab,
                                         observer=oname, W_s=W_s, W_frames=w, target=tname,
                                         n_windows=len(e), rho=round(rho, 4), p_event=round(float(e.mean()), 4),
                                         H_E=round(binary_entropy(float(e.mean())), 4),
                                         MI_plugin=round(mi, 5), MI_null=round(null, 5),
                                         MI_corr=round(corr, 5), p_value=round(pval, 4)))
        print(f"[{gi + 1}/{len(grid)}] {fam} {lab}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    df["informative"] = (df.dataset + "/" + df.video).isin(inform).astype(int)
    df.to_csv(out / "P0_mi_oracle.csv", index=False)

    # ---------------- summary with bootstrap CI over videos
    srows = []
    keys = ["family", "param", "observer", "W_s", "target"]
    # "evaluable": rows where the target is not degenerate and windows are enough for a plug-in
    # estimate (H(E_W) >= 0.5 bit, >= 30 windows). Defined per (video, W, target), not per video.
    ev_mask = (df.H_E >= 0.5) & (df.n_windows >= 30)
    for sset, sub in (("all", df), ("informative", df[df.informative == 1]), ("evaluable", df[ev_mask])):
        for k, g in sub.groupby(keys):
            x = g["MI_corr"].to_numpy()
            bs = rng.choice(x, (mcfg["n_boot"], len(x))).mean(1)
            srows.append(dict(zip(keys, k), set=sset, n_videos=len(g), rho_mean=round(g.rho.mean(), 4),
                              H_E_mean=round(g.H_E.mean(), 4), MI_corr_mean=round(x.mean(), 5),
                              ci_lo=round(np.quantile(bs, 0.025), 5), ci_hi=round(np.quantile(bs, 0.975), 5),
                              MI_corr_median=round(float(np.median(x)), 5),
                              frac_p_lt_0_05=round(float((g.p_value < 0.05).mean()), 3),
                              MI_norm_mean=round(float((g.MI_corr / g.H_E.where(g.H_E > 0)).mean()), 4)))
    summ = pd.DataFrame(srows)
    summ.to_csv(out / "P0_mi_summary.csv", index=False)
    plot(summ, out)
    print("wrote", out / "P0_mi_oracle.csv", out / "P0_mi_summary.csv")


def plot(summ: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [("content_aware_pure", "content-aware (pure threshold)", "#c0392b", "o-"),
            ("content_aware_shield", "content-aware + shield (L=5)", "#e67e22", "s-"),
            ("periodic", "periodic", "#2c3e50", "^-"),
            ("rate_limited", "rate-limited K/25 + floor P=5", "#2980b9", "D-"),
            ("privacy_certified", "rate-limited K=5 + dummy λ_d", "#27ae60", "v-")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, tgt, ttl in zip(axes, ["E_pres", "E_onset"], ["A1: event in window", "A2: onset in window"]):
        s = summ[(summ.set == "evaluable") & (summ.observer == "oracle") & (summ.W_s == 1) & (summ.target == tgt)]
        for fam, name, col, sty in fams:
            g = s[s.family == fam].sort_values("rho_mean")
            ax.plot(g.rho_mean, g.MI_corr_mean, sty, color=col, label=name, ms=4)
            ax.fill_between(g.rho_mean, g.ci_lo, g.ci_hi, color=col, alpha=0.15)
        ax.set_xlabel("open rate ρ (energy proxy)")
        ax.set_ylabel("I(E; N_W) bits/window (bias-corrected)")
        ax.set_title(f"{ttl}, W = 1 s, evaluable videos (n={int(s.n_videos.max())})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "P0_mi_oracle.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
