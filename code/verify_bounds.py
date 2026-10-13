"""verify_bounds.py — T10: theorem bounds vs measured numbers (PREREG_39 §8).

  thm2   aligned-block SYN (L = 5): p1_hat, p0_hat, rho0_hat, I(Z;B) measured vs f(pi, L rho0) for
         G8 (content-aware OR refresh P_r, no reuse cap), G4 periodic P in {5, 8}, and G8* (oracle-certified:
         one open at every event onset + Bernoulli(q) opens elsewhere; added 09-10, see PREREG §9);
         Corollary 2.1(a) check with unaligned onsets and a known candidate time.  Fig. 1 (trilemma).
  thm1   Thm 1(c): I(E; N_W) measured for G5 (rate-limited, sliding bucket) vs log2(K_W + 1), A1 and a multi-bit
         A3 target (onset count in W).
  thm3   G7 grid: rho, eps0, Delta(D) and eps for D in {5, 25, 125} frames, measured I (A1, W = 1 s, oracle, from the
         MI table), mean latency and short-event (D < L) miss rate vs an iso-energy periodic schedule; closed-form
         Pareto (K = inf).
  thm3seeds (P2 T17) thm3 for seeds 42-46 -> results/P2_thm3_seed_sweep.csv (iso-periodic comparator spread)
  thm2v2 (P1b T14, post-hoc) thm2 aligned-block check with the SYN v2 score model -> results/P1b_bounds_thm2_synv2.csv
Outputs: results/P1_bounds_thm{1,2,3}.csv, results/P1_fig1_trilemma.png, results/P1_fig_thm3_pareto.png
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import pandas as pd

import gates as G
import mi_estimators as M
from data_sparse import load_corpus, synth_score
from schedules import load_config, results_dir

CFG = load_config()
L = CFG["certificate_L"]
OUT = results_dir(CFG)
MODEL = json.loads((OUT / "P1_synthetic_model.json").read_text(encoding="utf-8"))
PIS = CFG["synthetic"]["pi_sweep"]
SEED = 42          # set by --seed; all RNG streams are default_rng([SEED, ...]) — no hash(), no PYTHONHASHSEED


def lab_id(lab: str) -> int:
    """Deterministic integer id of a gate label (replaces hash(lab), which is salted per process)."""
    return zlib.crc32(lab.encode("utf-8"))


def f_bound(pi, x):
    x = min(max(x, 0.0), 1.0)
    return M.h2(pi + (1 - pi) * x) - (1 - pi) * M.h2(x)


def i_general(pi, p1, p0):
    return M.h2(pi * p1 + (1 - pi) * p0) - pi * M.h2(p1) - (1 - pi) * M.h2(p0)


def mi_binary(z, b):
    """Plug-in I(Z;B) for two binary vectors (bits)."""
    pz, pb = z.mean(), b.mean()
    out = 0.0
    for zz in (0, 1):
        for bb in (0, 1):
            pj = np.mean((z == zz) & (b == bb))
            if pj > 0:
                out += pj * np.log2(pj / ((pz if zz else 1 - pz) * (pb if bb else 1 - pb)))
    return out


# ------------------------------------------------------------------ Thm 2
def aligned_video(pi_b, n_blocks, D_pool, rng):
    """Blocks of length L: Z_j = 1 event starts at block start (D >= L), 0 quiet block, -1 inside an event."""
    z = np.full(n_blocks, 0, np.int8)
    e = np.zeros(n_blocks * L, np.int8)
    j = 0
    while j < n_blocks:
        if rng.random() < pi_b:
            d = int(rng.choice(D_pool))
            nb = int(np.ceil(d / L))
            z[j] = 1
            z[j + 1:j + nb] = -1
            e[j * L:min((j + nb) * L, len(e))] = 1     # event fills whole blocks (aligned)
            j += nb
            if j < n_blocks:
                z[j] = 0                              # one quiet block after each event (keeps starts separated)
                j += 1
        else:
            j += 1
    return z, e


def oracle_certified(e, q, rng):
    on = np.zeros(len(e), np.int8)
    on[1:] = (e[1:] == 1) & (e[:-1] == 0)
    on[0] = e[0]
    off = ((rng.random(len(e)) < q) & (e == 0)).astype(np.int8)
    return np.maximum(on, off)


def thm2_gates():
    g = [("G8_ca_refresh", f"Pr={P},tau={t}", dict(P_r=P, tau=t)) for P in (8, 15, 30) for t in (0.35, 0.55, 0.72)]
    g += [("G4_periodic", f"P={P}", dict(P=P)) for P in (5, 8)]
    g += [("G8star_oracle_cert", f"q={q}", dict(q=q)) for q in (0.0, 0.02, 0.05, 0.1, 0.15, 0.19)]
    return g


def run_thm2(model=None, tag=""):
    """tag == "" -> P1 outputs (+ Cor 2.1a, Fig. 1); tag != "" -> only the aligned-block table, suffixed (P1b)."""
    model = model or MODEL
    D = np.asarray(model["durations"])
    D_pool = D[D >= L]
    n_vid, n_blocks = CFG["synthetic"]["n_videos"], CFG["synthetic"]["T"] // L
    rows = []
    for pi_b in PIS:
        vids = []
        for v in range(n_vid):
            rng = np.random.default_rng([SEED, 2, int(pi_b * 1000), v])
            z, e = aligned_video(pi_b, n_blocks, D_pool, rng)
            vids.append((z, e, synth_score(e, model, rng), rng))
        for fam, lab, p in thm2_gates():
            per = []
            for vi, (z, e, s, _) in enumerate(vids):
                rng = np.random.default_rng([SEED, 3, int(pi_b * 1000), vi, lab_id(lab)])
                if fam == "G8star_oracle_cert":
                    a = oracle_certified(e, p["q"], rng)
                else:
                    a = G.make(s, np.ones(len(s), np.int8), fam, p, rng, L)
                B = a.reshape(-1, L).max(1)
                k = z >= 0
                per.append(dict(z=z[k], B=B[k], opens0=int(a[e == 0].sum()), frames0=int((e == 0).sum()),
                                q_frames=int(((z == 0).repeat(L)).sum()),
                                q_opens=int(a[(z == 0).repeat(L)].sum())))
            zz = np.concatenate([d["z"] for d in per]); BB = np.concatenate([d["B"] for d in per])
            pi_hat = zz.mean()
            p1 = BB[zz == 1].mean() if (zz == 1).any() else np.nan
            p0 = BB[zz == 0].mean()
            rho0 = sum(d["q_opens"] for d in per) / sum(d["q_frames"] for d in per)   # rate in quiet blocks
            I = mi_binary(zz, BB)
            boot = []
            rng = np.random.default_rng([SEED, 4, int(pi_b * 1000)])
            for _ in range(500):
                idx = rng.integers(0, len(per), len(per))
                z2 = np.concatenate([per[i]["z"] for i in idx]); B2 = np.concatenate([per[i]["B"] for i in idx])
                boot.append(mi_binary(z2, B2))
            fb = f_bound(pi_hat, L * rho0) if L * rho0 < 1 else 0.0
            rows.append(dict(pi_target=pi_b, pi_block=pi_hat, family=fam, param=lab, p1_hat=p1, p0_hat=p0,
                             rho0_hat=rho0, L_rho0=L * rho0, p0_le_Lrho0=bool(p0 <= L * rho0 + 1e-12),
                             I_meas=I, I_lo=np.quantile(boot, 0.025), I_hi=np.quantile(boot, 0.975),
                             f_bound=fb, I_general=i_general(pi_hat, p1, p0),
                             certified=bool(p1 == 1.0),
                             bound_holds=bool(I >= fb - 1e-9) if p1 == 1.0 else np.nan))
    df = pd.DataFrame(rows)
    if tag:
        df.to_csv(OUT / f"P1b_bounds_thm2_{tag}.csv", index=False)
        print(df.round(4).to_string())
        return df
    # Corollary 2.1(a): unaligned onsets, known candidate time tau0 (random phase)
    rows2 = []
    for pi_b in (0.05, 0.2):
        for q in (0.02, 0.05, 0.1, 0.15):
            zs, bs, n0, nf = [], [], 0, 0
            for v in range(n_vid):
                rng = np.random.default_rng([SEED, 5, int(pi_b * 1000), v, int(q * 1000)])
                T = CFG["synthetic"]["T"]
                n_cand = T // (4 * L)
                quiet = np.zeros(T, np.int8)
                a0 = ((rng.random(T) < q)).astype(np.int8)              # quiet-world schedule
                n0 += int(a0.sum()); nf += T
                t0 = rng.integers(0, T - L + 1, n_cand)
                z = (rng.random(n_cand) < pi_b).astype(np.int8)
                # under H1 the certificate forces an open in [t0, t0+L-1]; under H0 we read the quiet schedule
                b = np.array([1 if zi else int(a0[t:t + L].max()) for zi, t in zip(z, t0)], np.int8)
                zs.append(z); bs.append(b)
            z, b = np.concatenate(zs), np.concatenate(bs)
            rho0 = n0 / nf
            T = CFG["synthetic"]["T"]
            rows2.append(dict(check="Cor2.1a_random_phase", pi=z.mean(), q=q, rho0_hat=rho0,
                              p0_hat=b[z == 0].mean(), L_rho0_adj=L * rho0 * T / (T - L + 1),
                              I_meas=mi_binary(z, b), f_bound=f_bound(z.mean(), L * rho0 * T / (T - L + 1))))
    df2 = pd.DataFrame(rows2)
    df.to_csv(OUT / "P1_bounds_thm2.csv", index=False)
    df2.to_csv(OUT / "P1_bounds_thm2_cor21a.csv", index=False)
    fig1(df)
    print(df.round(4).to_string())
    print(df2.round(4).to_string())
    return df


def fig1(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # (a) f(pi = 0.1, L rho0) over (L, rho0); zero-leakage frontier rho0 = 1/L
    Ls = np.arange(2, 31)
    rhos = np.linspace(0.005, 0.6, 240)
    Z = np.array([[f_bound(0.1, l * r) if l * r < 1 else 0.0 for r in rhos] for l in Ls])
    im = ax[0].pcolormesh(rhos, Ls, Z, shading="auto", cmap="magma_r")
    ax[0].plot(1 / Ls, Ls, "c-", lw=2, label=r"$\rho_0 = 1/L$ (periodic $P=L$, $I=0$)")
    ax[0].set_xlabel(r"off-event open rate $\rho_0$ (energy)")
    ax[0].set_ylabel(r"certified latency $L$ (frames)")
    ax[0].set_title(r"(a) forced leakage $f(\pi{=}0.1, L\rho_0)$ [bit/block]", fontsize=9)
    fig.colorbar(im, ax=ax[0])
    ax[0].legend(fontsize=7, loc="upper right")
    # (b) bound curves vs measured points (L = 5)
    x = np.linspace(0, 1, 200)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(PIS)))
    for pi, col in zip(PIS, cols):
        s = df[(df.pi_target == pi)]
        pib = float(s.pi_block.iloc[0])
        ax[1].plot(x, [f_bound(pib, xx) for xx in x], "-", color=col, lw=1.2, label=fr"$f(\pi_b{{=}}{pib:.3f},\cdot)$")
        for fam, mk in (("G8star_oracle_cert", "o"), ("G8_ca_refresh", "x"), ("G4_periodic", "s")):
            g = s[s.family == fam]
            ax[1].scatter(np.clip(g.L_rho0, 0, 1.05), g.I_meas, marker=mk, color=col, s=18)
    ax[1].axvline(1.0, color="grey", ls=":", lw=1)
    ax[1].set_xlabel(r"$L\rho_0$  (L = 5)")
    ax[1].set_ylabel(r"$I(Z;B)$ bit/block")
    ax[1].set_title("(b) bound vs measured (o oracle-certified, x CA+refresh, s periodic)", fontsize=8)
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "P1_fig1_trilemma.png", dpi=160)


# ------------------------------------------------------------------ Thm 1(c)
def run_thm1():
    rows = []
    for corpus in ("C121", "SYN_pi0.1", "VIRAT"):
        clips = load_corpus(corpus)
        for K in (3, 5, 8, 12):
            for W_s in CFG["mi"]["windows_s"]:
                for i, c in enumerate(clips):
                    w = int(round(W_s * c.fps))
                    a = G.make(c.score, c.valid, "G5_rate_limited", dict(K=K), np.random.default_rng([SEED, 7, i, K]), L)
                    m = c.n // w
                    if m < 30:
                        continue
                    N = a[: m * w].reshape(m, w).sum(1)
                    E = c.e[: m * w].reshape(m, w)
                    on = np.zeros(c.n, np.int8)
                    on[1:] = (c.e[1:] == 1) & (c.e[:-1] == 0)
                    cnt = on[: m * w].reshape(m, w).sum(1)
                    KW = K * int(np.ceil(w / 25))
                    rng = np.random.default_rng([SEED, 8, i, K, W_s])
                    a1 = M.plugin(E.max(1).astype(np.int8), np.clip(N / w, 0, 1), rng, 200)[2]
                    # multi-bit target: onset count (capped at 7), plug-in with permutation correction
                    y = np.minimum(cnt, 7)
                    xb = M.bin_rate(np.clip(N / w, 0, 1))
                    def mi_xy(yy):
                        j = np.zeros((8, M.N_BINS))
                        np.add.at(j, (yy, xb), 1)
                        j /= j.sum()
                        px, py = j.sum(0), j.sum(1)
                        nz = j > 0
                        return float((j[nz] * np.log2(j[nz] / np.outer(py, px)[nz])).sum())
                    obs_mi = mi_xy(y)
                    null = np.mean([mi_xy(rng.permutation(y)) for _ in range(50)])
                    py = np.bincount(y, minlength=8) / len(y)
                    Hy = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
                    rows.append(dict(corpus=corpus, video=c.key, K=K, W_s=W_s, K_W=KW, bound=np.log2(KW + 1),
                                     I_A1=a1, H_A1=M.h2(float(E.max(1).mean())), I_count=max(0.0, obs_mi - null),
                                     H_count=Hy))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "P1_bounds_thm1_per_video.csv", index=False)
    s = df.groupby(["corpus", "K", "W_s"]).agg(n_videos=("video", "size"), bound=("bound", "first"),
                                                I_A1=("I_A1", "mean"), H_A1=("H_A1", "mean"),
                                                I_count=("I_count", "mean"), H_count=("H_count", "mean"),
                                                max_I_count=("I_count", "max")).reset_index()
    s["bound_minus_H_count"] = s.bound - s.H_count
    s.to_csv(OUT / "P1_bounds_thm1.csv", index=False)
    print(s.round(3).to_string())


# ------------------------------------------------------------------ Thm 3
def eps0(p1, lam):
    return max(np.log(p1 / lam), np.log((1 - lam) / (1 - p1)))


def delta(K, Wb, D, h=0):
    F = D + h
    if K >= Wb:
        return F
    nw = int(np.ceil((F - 1) / Wb)) + 1
    return int(min(2 * K * nw, 2 * F))


def latency_stats(a, e):
    """Per event: latency (frames to first open at/after onset, capped at L), and miss for D < L."""
    on = np.flatnonzero(np.diff(np.r_[0, e]) == 1)
    off = np.flatnonzero(np.diff(np.r_[e, 0]) == -1)
    nxt = np.full(len(a) + 1, len(a))
    idx = np.flatnonzero(a)
    nxt[idx] = idx
    for t in range(len(a) - 1, -1, -1):
        nxt[t] = min(nxt[t], nxt[t + 1])
    lat, miss_short = [], []
    for s, t_end in zip(on, off):
        d = t_end - s + 1
        lat.append(nxt[s] - s)
        if d < L:
            miss_short.append(int(nxt[s] > t_end))
    return np.array(lat, float), np.array(miss_short, float)


def periodic_rate(n, rho, phase):
    """Oblivious schedule with exact long-run rate rho (Bresenham)."""
    t = np.arange(n)
    return (np.floor((t + 1) * rho + phase) - np.floor(t * rho + phase)).astype(np.int8)


def run_thm3(write=True):
    mi = pd.read_csv(OUT / "P1_mi_table.csv") if (OUT / "P1_mi_table.csv").exists() else None
    rows = []
    for corpus in ("SYN_pi0.05", "SYN_pi0.1", "C121", "VIRAT"):
        clips = load_corpus(corpus)
        cfgs = [("G7_rr", f"K={G.kname(K)},p1={p1},lam={lam}", dict(K=K, p1=p1, lam=lam)) for K, p1, lam in G.G7_GRID]
        cfgs += [("G4_periodic", "P=5", dict(P=5)), ("G1_ca_shield", "tau=0.55", dict(tau=0.55)),
                 ("G3_deployed", "hyst", {})]
        for fam, lab, p in cfgs:
            lat, miss, rho, latp, missp = [], [], [], [], []
            for i, c in enumerate(clips):
                rng = np.random.default_rng([SEED, 9, i, lab_id(lab)])
                a = G.make(c.score, c.valid, fam, p, rng, L)
                la, ms = latency_stats(a, c.e)
                lat.append(la); miss.append(ms); rho.append(a.mean())
                ap = periodic_rate(c.n, max(a.mean(), 1e-6), rng.random())
                la, ms = latency_stats(ap, c.e)
                latp.append(la); missp.append(ms)
            r = dict(corpus=corpus, family=fam, param=lab, rho=float(np.mean(rho)),
                     mean_latency=float(np.concatenate(lat).mean()),
                     p95_latency=float(np.quantile(np.concatenate(lat), 0.95)),
                     miss_short=float(np.concatenate(miss).mean()) if sum(len(m) for m in miss) else np.nan,
                     iso_periodic_mean_latency=float(np.concatenate(latp).mean()),
                     iso_periodic_miss_short=float(np.concatenate(missp).mean()) if sum(len(m) for m in missp) else np.nan)
            if fam == "G7_rr":
                e0 = eps0(p["p1"], p["lam"])
                r.update(eps0=e0, **{f"Delta_D{D}": delta(p["K"], 25, D) for D in (5, 25, 125)},
                         **{f"eps_D{D}": delta(p["K"], 25, D) * e0 for D in (5, 25, 125)})
            if mi is not None:
                s = mi[(mi.corpus == corpus) & (mi.family == fam) & (mi.param == lab) & (mi.observer == "oracle")
                       & (mi.W_s == 1) & (mi.target == "A1") & (mi.estimator == "plugin_corr")]
                for tier in ("all", "evaluable"):
                    t = s[s.tier == tier]
                    r[f"I_A1_W1_{tier}"] = float(t["mean"].iloc[0]) if len(t) else np.nan
            rows.append(r)
        print(f"thm3 {corpus} done", flush=True)
    df = pd.DataFrame(rows)
    if not write:
        return df
    df.to_csv(OUT / "P1_bounds_thm3.csv", index=False)
    fig_thm3(df)
    print(df.round(3).to_string())
    return df


def run_thm3_seeds(seeds=(42, 43, 44, 45, 46)):
    """P2 T17: seed sensitivity of the Thm 3 iso-energy comparison (gate RR draws + one periodic phase per clip)."""
    global SEED
    keep, out = SEED, []
    for s in seeds:
        SEED = s
        df = run_thm3(write=False)
        df["seed"] = s
        out.append(df)
    SEED = keep
    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT / "P2_thm3_seed_sweep.csv", index=False)
    print(df.groupby(["corpus", "family"]).size())


def fig_thm3(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # (a) closed-form Pareto, K = inf: min energy rho* vs eps0 for target p1, d_bar = 0.05
    e = np.linspace(0.05, 5, 200)
    for p1, col in zip((0.5, 0.7, 0.9), ("#1b9e77", "#d95f02", "#7570b3")):
        for dbar, ls in ((0.05, "-"), (0.5, "--")):
            lam = np.maximum(np.maximum(p1 * np.exp(-e), 1 - (1 - p1) * np.exp(e)), 0)
            rho = 1 / L + (1 - 1 / L) * (lam * (1 - dbar) + p1 * dbar)
            ax[0].plot(e, rho, ls, color=col, label=fr"$p_1={p1}$, $\bar d={dbar}$")
    ax[0].axhline(1 / L, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel(r"$\varepsilon_0$ (per-frame privacy loss)")
    ax[0].set_ylabel(r"minimum open rate $\rho^*$")
    ax[0].set_title(r"(a) Thm 3(v) closed-form Pareto, $K=\infty$, L = 5", fontsize=9)
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)
    # (b) measured: mean latency vs rho, RR vs iso-energy periodic (SYN pi = 0.05)
    s = df[(df.corpus == "SYN_pi0.05") & (df.family == "G7_rr")]
    sc = ax[1].scatter(s.rho, s.mean_latency, c=s.eps0, cmap="viridis", s=25, label="RR gate (colour = eps0)")
    ax[1].scatter(s.rho, s.iso_periodic_mean_latency, marker="x", color="grey", s=20, label="iso-energy periodic")
    fig.colorbar(sc, ax=ax[1])
    ax[1].set_xlabel(r"open rate $\rho$")
    ax[1].set_ylabel("mean detection latency (frames)")
    ax[1].set_title(r"(b) SYN $\pi$=0.05: RR vs iso-energy periodic", fontsize=9)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "P1_fig_thm3_pareto.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["thm2", "thm1", "thm3"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    SEED = args.seed
    for part in args.parts:
        {"thm2": run_thm2, "thm1": run_thm1, "thm3": run_thm3,
         "fig1": lambda: fig1(pd.read_csv(OUT / "P1_bounds_thm2.csv")),
         "thm3seeds": run_thm3_seeds,
         "thm2v2": lambda: run_thm2(json.loads((OUT / "P1b_synthetic_model_v2.json").read_text(encoding="utf-8")),
                                    "synv2")}[part]()
