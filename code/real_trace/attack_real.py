"""attack_real.py — T22b Task 2: presence (A1) and activity (A3) attacks on the real INA219 trace.

POST HOC ("real-plug validation", PREREG_39 §9). Same window features, targets and classifiers as code/attacks.py
(window_feats / window_targets, GBM + logistic, HGB + ridge); no PREREG threshold is changed.

Views of the SAME G3 schedule, same windows (W in {1, 2, 5} s of video time):
  oracle     a_t exactly
  sim_plug   the simulated 1-Hz smart plug of attacks.py (f_s 1 Hz, sigma 0.05 W, quant 0.01 W, P_base 5 W)
  real_1hz   real trace averaged per second of VIDEO time (20 samples at 25 fps, 24 at 30 fps), inverted to an
             open rate with the device constants of calib.json -> apples-to-apples with sim_plug  [headline]
  real_10hz  every real 10-Hz sample, interpolated to frames (20-24 samples per video second because the replay is
             2.0x / 2.4x slower than video time) -> more samples per video second than any
             1-Hz plug; reported for reference (it is not uniformly better for the trained attacker)
Splits:
  within     per video: first half of the windows train, second half test (blizzard, VIRAT)
  cross      train on all of blizzard -> test on all of VIRAT, and the reverse
  smoke      LASIESTA O_RA_02 (375 frames): within split + blizzard -> O_RA_02; not in the main table
Uncertainty (GUM, C6): 5 seeds (seed = observation noise of sim_plug + bootstrap draws) x non-overlapping 10-s block
bootstrap of test windows (B = 1000). u_c = sqrt(mean_seed(sd_boot)^2 + sd_seed(point)^2), U = 2 u_c (k = 2).
MI: plug-in on per-window rate, corrected by a 200-shuffle permutation null, per video over all windows; U from a
10-s block bootstrap (B = 200, 100 shuffles each) x 5 seeds.

Outputs: results/real_trace/P3_real_attack.csv, P3_real_mi.csv, real_numbers.json (-> numbers.json real.* via
code/p1_report.py), figures P3_fig_real_trace.png, P3_fig_real_vs_sim_auc.png.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import attacks as A  # noqa: E402
import mi_estimators as M  # noqa: E402
from align import OUT, clip_of, selection  # noqa: E402
from data_sparse import g3  # noqa: E402

SEEDS = range(5)
W_LIST = (1, 2, 5)
B_AUC, B_MI, BLOCK_S = 1000, 200, 10
VIEWS = ("oracle", "sim_plug", "real_1hz", "real_10hz")
SIM_SPEC = A.OBS_HEAD["power_head"]          # ("power", 1.0, 0.05)
MAIN = ("cdnet_blizzard", "virat_S_000002")
SMOKE = "lasiesta_O_RA_02"


# ------------------------------------------------------------------ data
class Video:
    def __init__(self, tag, dataset, video, calib):
        self.tag = tag
        c = clip_of(dataset, video)
        self.c, self.fps, self.e = c, c.fps, c.e
        self.a = g3(c).astype(np.int8)
        al = pd.read_csv(OUT / f"aligned_{tag}.csv")
        self.al = al
        x_s = (al.I_mA.to_numpy() - calib["I_idle_mA"]) / calib["dI_frame_mA"]   # per-sample open estimate
        f = int(round(self.fps))
        # real_1hz: mean over the samples of each video second -> open rate of that second
        sec = (al.frame.to_numpy() // f)
        n_sec = self.c.n // f
        r1 = np.zeros(n_sec)
        for j in range(n_sec):
            m = sec == j
            r1[j] = max(0.0, x_s[m].mean()) if m.any() else 0.0
        x1 = np.zeros(self.c.n)
        x1[: n_sec * f] = np.repeat(r1, f)
        self.real_1hz = (x1, 0.5 / f)
        # real_10hz: sample values interpolated over frame index
        self.real_10hz = (np.clip(np.interp(np.arange(self.c.n), al.frame.to_numpy(), x_s), 0.0, None), 0.5)

    def view(self, name, seed):
        if name == "oracle":
            return self.a.astype(np.float64), 0.5
        if name == "sim_plug":
            return A.observe(self.a, self.fps, SIM_SPEC, A.seed("T22b", self.tag, seed))
        return getattr(self, name)

    def windows(self, name, seed, W_s):
        x, thr = self.view(name, seed)
        w = int(round(W_s * self.fps))
        return A.window_feats(x, w, thr), A.window_targets(self.e, w, self.fps), x


# ------------------------------------------------------------------ uncertainty
def block_ids(n, W_s):
    b = max(1, int(round(BLOCK_S / W_s)))
    return np.arange(n) // b


def boot_idx(blocks, rng):
    ub = np.unique(blocks)
    pick = rng.choice(ub, len(ub), replace=True)
    return np.concatenate([np.flatnonzero(blocks == k) for k in pick])


def auc_boot(y, s, W_s, rng, B=B_AUC):
    if y.min() == y.max():
        return np.nan, np.array([])
    blocks = block_ids(len(y), W_s)
    out = []
    for _ in range(B):
        i = boot_idx(blocks, rng)
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.array(out)


def combine(points, boots):
    """GUM combination over seeds: value = mean point; u_c^2 = mean(sd_boot)^2 + sd(points)^2; U = 2 u_c."""
    p = np.array(points, float)
    sd_b = np.array([b.std(ddof=1) if len(b) > 1 else np.nan for b in boots])
    u_boot = float(np.nanmean(sd_b)) if np.isfinite(sd_b).any() else 0.0   # A3 rows: seed spread only
    u_seed = float(p.std(ddof=1)) if len(p) > 1 else 0.0
    allb = np.concatenate([b for b in boots if len(b)]) if any(len(b) for b in boots) else np.array([np.nan])
    return dict(value=float(p.mean()), u_boot=u_boot, u_seed=u_seed, U=2.0 * float(np.hypot(u_boot, u_seed)),
                ci_lo=float(np.nanquantile(allb, 0.025)), ci_hi=float(np.nanquantile(allb, 0.975)))


# ------------------------------------------------------------------ attacks
def fit_predict(Xtr, ytr, Xte, model):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), float(ytr[0]))
    clf = A.gbm_clf() if model == "gbm" else A.logit_clf()
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def reg_predict(Xtr, ytr, Xte, model):
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=42) if model == "gbm" \
        else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return reg.fit(Xtr, ytr).predict(Xte)


def run_split(split, train, test, vids, rows):
    """train/test: (tag, part) with part in {'first', 'second', 'all'}."""
    for W_s in W_LIST:
        for view in VIEWS:
            pts = {}
            for seed in SEEDS:
                def take(tag, part):
                    X, T, _ = vids[tag].windows(view, seed, W_s)
                    h = len(X) // 2
                    sl = slice(0, h) if part == "first" else slice(h, None) if part == "second" else slice(None)
                    return X[sl], {k: v[sl] for k, v in T.items()}
                Xtr, Ttr = take(*train)
                Xte, Tte = take(*test)
                rng = A.seed("T22b-boot", split, train, test, view, W_s, seed)
                for model in ("gbm", "logit"):
                    p = fit_predict(Xtr, Ttr["A1"], Xte, model)
                    # A constant prediction is not an attack result: HGB (min_samples_leaf 20) cannot split on the
                    # 24 training windows of the W = 5 s within-video split. Record it as not evaluable (NaN).
                    if np.ptp(p) == 0:
                        pts.setdefault(("A1", model, "AUC"), []).append((np.nan, np.array([])))
                        continue
                    auc, bs = auc_boot(Tte["A1"], p, W_s, rng)
                    pts.setdefault(("A1", model, "AUC"), []).append((auc, bs))
                for model in ("gbm", "ridge"):
                    po = reg_predict(Xtr, Ttr["occ"], Xte, model)
                    pr = reg_predict(Xtr, Ttr["rate"], Xte, model)
                    r2 = r2_score(Tte["occ"], po) if np.var(Tte["occ"]) > 0 else np.nan
                    pts.setdefault(("A3occ", model, "R2"), []).append((r2, np.array([])))
                    pts.setdefault(("A3rate", model, "MAE_events_per_min"), []).append(
                        (mean_absolute_error(Tte["rate"], pr), np.array([])))
            n_tr, n_te = len(Xtr), len(Xte)
            for (target, model, metric), lst in pts.items():
                vals = [v for v, _ in lst]
                if all(np.isnan(vals)):
                    continue
                st = combine(vals, [b for _, b in lst])
                rows.append(dict(split=split, train=f"{train[0]}:{train[1]}", test=f"{test[0]}:{test[1]}",
                                 view=view, W_s=W_s, target=target, model=model, metric=metric, **st,
                                 n_train_windows=n_tr, n_test_windows=n_te,
                                 p_event_test=float(Tte["A1"].mean()), n_seeds=len(vals)))
        print(f"  {split} {train}->{test} W={W_s}s done", flush=True)


def run_mi(vids, tags, rows):
    for tag in tags:
        v = vids[tag]
        for W_s in W_LIST:
            for view in VIEWS:
                pts, boots, p0 = [], [], np.nan
                for seed in SEEDS:
                    _, T, x = v.windows(view, seed, W_s)
                    w = int(round(W_s * v.fps))
                    m = v.c.n // w
                    rate = np.clip(x[: m * w].reshape(m, w).sum(1) / w, 0, 1)
                    e = T["A1"]
                    rng = A.seed("T22b-mi", tag, view, W_s, seed)
                    mi, null, corr, pval = M.plugin(e, rate, rng, 200)
                    blocks = block_ids(m, W_s)
                    bs = []
                    for _ in range(B_MI):
                        i = boot_idx(blocks, rng)
                        bs.append(M.plugin(e[i], rate[i], rng, 100)[2])
                    pts.append(corr)
                    p0 = pval if seed == 0 else p0
                    boots.append(np.array(bs))
                st = combine(pts, boots)
                rows.append(dict(video=tag, view=view, W_s=W_s, target="A1", estimator="plugin_corr", **st,
                                 n_windows=int(m), p_event=float(e.mean()), H_E=M.h2(float(e.mean())),
                                 p_value_seed0=p0, n_seeds=len(pts)))
        print(f"  MI {tag} done", flush=True)


# ------------------------------------------------------------------ verdicts + numbers
def verdicts(att: pd.DataFrame) -> pd.DataFrame:
    a = att[(att.target == "A1") & (att.metric == "AUC")]
    out = []
    for (split, tr, te, W, model), g in a.groupby(["split", "train", "test", "W_s", "model"]):
        g = g.set_index("view")
        if not set(VIEWS) <= set(g.index):
            continue
        r, s = g.loc["real_1hz"], g.loc["sim_plug"]
        d = r.value - s.value
        Ud = float(np.hypot(r.U, s.U))
        verdict = "consistent" if abs(d) <= r.U else ("lower" if d < 0 else "higher")
        out.append(dict(split=split, train=tr, test=te, W_s=W, model=model, auc_oracle=g.loc["oracle"].value,
                        auc_sim=s.value, U_sim=s.U, auc_real_1hz=r.value, U_real_1hz=r.U,
                        auc_real_10hz=g.loc["real_10hz"].value, U_real_10hz=g.loc["real_10hz"].U,
                        diff_real_minus_sim=d, U_diff=Ud, verdict_sim_in_real_U=verdict,
                        within_U_diff=bool(abs(d) <= Ud)))
    return pd.DataFrame(out)


def real_numbers(att, mi, ver, calib) -> dict:
    N = {}
    c = {k: v for k, v in calib.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    N["real.calib"] = c
    N["real.run"] = {"samples": int(calib["samples_total"]), "n_videos": 3, "fps_play": calib["fps_play"],
                     "busy_ms": calib["busy_ms"], "sample_hz": calib["sample_hz"],
                     "dilation_25": round(25 / calib["fps_play"], 1), "dilation_30": round(30 / calib["fps_play"], 1),
                     "n_seeds": len(SEEDS), "B_auc": B_AUC, "block_s": BLOCK_S}
    short = {"cdnet_blizzard": "blizzard", "virat_S_000002": "virat", "lasiesta_O_RA_02": "lasiesta"}
    a = att[att.target == "A1"]
    for r in a.itertuples():
        trn, tst = r.train.split(":")[0], r.test.split(":")[0]
        name = short[tst] if r.split in ("within", "smoke") and trn == tst else f"{short[trn]}2{short[tst]}"
        sfx = "" if r.model == "gbm" else f"_{r.model}"
        N[f"real.auc_{r.split}_{name}_{r.view}_W{r.W_s}{sfx}"] = {
            "value": round(r.value, 4), "ci": [round(r.ci_lo, 4), round(r.ci_hi, 4)], "U": round(r.U, 4),
            "n_test_windows": int(r.n_test_windows)}
    for r in mi.itertuples():
        N[f"real.mi_{short[r.video]}_{r.view}_W{r.W_s}"] = {
            "value": round(r.value, 4), "ci": [round(r.ci_lo, 4), round(r.ci_hi, 4)], "U": round(r.U, 4),
            "H_E": round(r.H_E, 4), "n_windows": int(r.n_windows)}
    for split in ("within", "cross"):
        for model in ("gbm", "logit"):
            w = ver[(ver.split == split) & (ver.model == model)]
            sfx = "" if model == "gbm" else f"_{model}"
            N[f"real.{split}_verdict{sfx}"] = {
                "n_consistent": int((w.verdict_sim_in_real_U == "consistent").sum()), "of": int(len(w)),
                "n_lower": int((w.verdict_sim_in_real_U == "lower").sum()),
                "n_higher": int((w.verdict_sim_in_real_U == "higher").sum()),
                "diff_min": round(float(w.diff_real_minus_sim.min()), 4),
                "diff_max": round(float(w.diff_real_minus_sim.max()), 4)}
    m = att[(att.split == "within") & (att.target == "A3occ") & (att.model == "gbm") & (att.view == "real_1hz")]
    N["real.a3occ_R2_real_1hz"] = {"min": round(float(m.value.min()), 3), "max": round(float(m.value.max()), 3)}
    return N


# ------------------------------------------------------------------ figures
def fig_trace(vids, calib):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = vids["cdnet_blizzard"]
    f = int(v.fps)
    SECONDS = 60
    e_sec = v.e[: (v.c.n // f) * f].reshape(-1, f).max(1)
    best, k0 = -1, 0
    for s in range(0, len(e_sec) - SECONDS):           # 60-s stretch with the most balanced presence
        p = e_sec[s:s + SECONDS].mean()
        sc = min(p, 1 - p) + 0.01 * np.abs(np.diff(e_sec[s:s + SECONDS])).sum()
        if sc > best:
            best, k0 = sc, s
    fr0, fr1 = k0 * f, (k0 + SECONDS) * f
    al = v.al[(v.al.frame >= fr0) & (v.al.frame < fr1)]
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax[0].plot(al.t_video - k0, al.I_mA, color="#c0392b", lw=0.6, label="INA219, 10 Hz (20 samples per video s)")
    x1, _ = v.real_1hz
    r1 = x1[fr0:fr1:f]
    ax[0].step(np.arange(SECONDS), calib["I_idle_mA"] + r1 * calib["dI_frame_mA"], where="post", color="#2c3e50",
               lw=1.2, label="mean per video second (1-Hz equivalent)")
    ax[0].axhline(calib["I_idle_mA"], color="grey", lw=0.6, ls=":")
    ax[0].set_ylabel("board current (mA)", fontsize=7)
    ax[0].legend(fontsize=6, loc="upper right")
    t = np.arange(fr0, fr1)
    tt = (t - fr0) / v.fps
    ax[1].fill_between(tt, 0, v.e[fr0:fr1] * 1.0, step="post", color="#bbb", lw=0, label="event (ground truth)")
    ax[1].vlines(tt[v.a[fr0:fr1] == 1], 0.15, 0.85, color="#c0392b", lw=0.4, label="gate opens $a_t$")
    ax[1].set_yticks([])
    ax[1].set_ylabel("schedule", fontsize=7)
    ax[1].set_xlabel("video time (s)", fontsize=8)
    ax[1].legend(fontsize=6, loc="upper right", ncol=2)
    for a_ in ax:
        a_.tick_params(labelsize=7)
    fig.suptitle(f"CDnet2014/blizzard, deployed gate replayed on an ESP32-S3: 60 s of the real shunt trace", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "P3_fig_real_trace.png", dpi=200)
    plt.close(fig)
    return k0


def fig_auc(att):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # GBM (primary attacker) only. At W = 5 s the within-video split has 24-30 training windows, too few for the GBM to
    # split; the logistic fallback is unstable there (oracle AUC below chance on VIRAT), so W = 5 s is left to the CSV.
    a = att[(att.split == "within") & (att.target == "A1") & (att.model == "gbm")]
    w_list = sorted(set(a.W_s))
    style = {"oracle": ("k", "o", "oracle $a_t$"), "sim_plug": ("#e67e22", "s", "simulated plug, 1 Hz"),
             "real_1hz": ("#c0392b", "D", "real INA219, 1 Hz-equiv."),
             "real_10hz": ("#8e44ad", "^", "real INA219, raw 10 Hz")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, tag, title in zip(axes, MAIN, ("CDnet2014/blizzard", "VIRAT_S_000002")):
        g = a[a.test.str.startswith(tag)]
        for j, view in enumerate(VIEWS):
            col, mk, lab = style[view]
            s = g[g.view == view].sort_values("W_s")
            xpos = np.arange(len(w_list)) + (j - 1.5) * 0.15
            lo = np.minimum(s.U, s.value - 0.0)
            hi = np.minimum(s.U, 1.0 - s.value)          # AUC <= 1: clip the drawn bar, not the value
            ax.errorbar(xpos, s.value, yerr=[lo, hi], fmt=mk, color=col, ms=4, capsize=2, lw=0.8, label=lab)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks(range(len(w_list)), [f"W = {w} s" for w in w_list], fontsize=7)
        ax.set_title(title + " (second half, trained on first half)", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("presence AUC (GBM), $\\pm U$ ($k$=2)", fontsize=7)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "P3_fig_real_vs_sim_auc.png", dpi=200)
    plt.close(fig)


def copy_figs() -> None:
    """Copy the two figures into the manuscript folder when it exists (it does not in the public artifact)."""
    figs = HERE.parent.parent / "submission_package" / "manuscript" / "figs"
    if figs.is_dir():
        for name in ("P3_fig_real_trace.png", "P3_fig_real_vs_sim_auc.png"):
            shutil.copy(OUT / name, figs / name)


def main() -> None:
    t0 = time.time()
    calib = json.loads((OUT / "calib.json").read_text(encoding="utf-8"))
    sel = selection()
    vids = {r.tag: Video(r.tag, r.dataset, r.video, calib) for r in sel.itertuples()}
    if "--figs-only" in sys.argv:                  # redraw figures from the saved tables
        fig_trace(vids, calib)
        fig_auc(pd.read_csv(OUT / "P3_real_attack.csv"))
        copy_figs()
        return
    rows = []
    for tag in MAIN:
        run_split("within", (tag, "first"), (tag, "second"), vids, rows)
    run_split("cross", (MAIN[0], "all"), (MAIN[1], "all"), vids, rows)
    run_split("cross", (MAIN[1], "all"), (MAIN[0], "all"), vids, rows)
    run_split("smoke", (SMOKE, "first"), (SMOKE, "second"), vids, rows)
    run_split("smoke", (MAIN[0], "all"), (SMOKE, "all"), vids, rows)
    att = pd.DataFrame(rows)
    att.to_csv(OUT / "P3_real_attack.csv", index=False, float_format="%.5f")
    mi_rows = []
    run_mi(vids, [*MAIN, SMOKE], mi_rows)
    mi = pd.DataFrame(mi_rows)
    mi.to_csv(OUT / "P3_real_mi.csv", index=False, float_format="%.5f")
    ver = verdicts(att)
    ver.to_csv(OUT / "P3_real_vs_sim.csv", index=False, float_format="%.4f")
    N = real_numbers(att, mi, ver, calib)
    k0 = fig_trace(vids, calib)
    N["real.fig_trace_start_s"] = int(k0)
    (OUT / "real_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    fig_auc(att)
    copy_figs()
    pd.set_option("display.width", 220)
    print(ver.round(3).to_string(index=False))
    print(mi[mi.video != SMOKE].pivot_table(index=["video", "W_s"], columns="view", values="value").round(3))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
