"""p2_fig_trace.py — P2 T18: illustrative trace figure for the manuscript (one C121 evaluable video).

Panels (shared time axis, 40 s): event indicator E_t; deployed-gate opens (G3); periodic schedule with the same open
rate; 1-Hz smart-plug power for both schedules (sigma = 0.05 W, as the headline observer).
Output: results/P2_fig_trace.png (+ copy into submission_package/manuscript/figs/).
"""
from __future__ import annotations

import shutil

import numpy as np

import gates as G
import mi_estimators as M
import observer as obs
from data_sparse import load_corpus
from schedules import load_config, results_dir
from verify_bounds import periodic_rate

CFG = load_config()
OUT = results_dir(CFG)
SECONDS = 40


def pick(clips):
    """Evaluable and informative video (A1, W = 1 s; deployed rate < 0.6) whose 40-s stretch has the most event onsets with quiet gaps."""
    best, score = None, -1
    for c in clips:
        w = int(round(c.fps))
        m = c.n // w
        a1 = c.e[: m * w].reshape(m, w).max(1)
        if M.h2(float(a1.mean())) < 0.5 or m < 30 or G.deployed(c.score, c.valid).mean() > 0.6:
            continue                          # evaluable AND informative (deployed open rate < 0.6)
        L = SECONDS * w
        for s in range(0, c.n - L, w * 5):
            seg = c.e[s:s + L]
            on = int((np.diff(np.r_[0, seg]) == 1).sum())
            q = 1 - seg.mean()
            if 0.3 < q < 0.8 and on * q > score:
                best, score = (c, s), on * q
    return best


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c, s0 = pick(load_corpus("C121"))
    fps = c.fps
    L = int(SECONDS * fps)
    a3 = G.deployed(c.score, c.valid)
    rng = np.random.default_rng(42)
    ap = periodic_rate(c.n, float(a3.mean()), rng.random())
    sl = slice(s0, s0 + L)
    t = np.arange(L) / fps
    ocfg = CFG["observer"]
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.6]))
    ax[0].fill_between(t, 0, c.e[sl], step="post", color="#555", lw=0)
    ax[0].set_ylabel("event", fontsize=7)
    ax[1].vlines(t[a3[sl] == 1], 0, 1, color="#c0392b", lw=0.5)
    ax[1].set_ylabel("deployed\nopens", fontsize=7)
    ax[2].vlines(t[ap[sl] == 1], 0, 1, color="#2c3e50", lw=0.5)
    ax[2].set_ylabel("periodic\nopens", fontsize=7)
    for a, col, lab in ((a3, "#c0392b", f"deployed (rho = {a3.mean():.2f})"), (ap, "#2c3e50", "periodic, same rho")):
        p = obs.power_trace(a, fps, ocfg, 1.0, 0.05, np.random.default_rng(7))
        k0 = s0 // int(round(fps))
        ax[3].step(np.arange(SECONDS), p[k0:k0 + SECONDS], where="post", color=col, lw=1.0, label=lab)
    ax[3].set_ylabel("plug power\n(W, 1 Hz)", fontsize=7)
    ax[3].set_xlabel("time (s)", fontsize=8)
    ax[3].legend(fontsize=6, loc="upper right")
    for a in ax[:3]:
        a.set_yticks([])
    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle(f"{c.dataset}/{c.video}: events, gate opens and what a 1-Hz smart plug sees", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "P2_fig_trace.png", dpi=200)
    shutil.copy(OUT / "P2_fig_trace.png", OUT.parent / "submission_package" / "manuscript" / "figs" / "P2_fig_trace.png")
    print(c.key, s0, round(float(a3.mean()), 3))


if __name__ == "__main__":
    main()
