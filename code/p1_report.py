"""p1_report.py — P1 figures + results/numbers.json (every number that may be printed in the paper, key-by-key).

Figures (results/):
  P1_fig_cap_tau.png        leakage vs tau (the cap/∩ shape): MI plug-in and A1 AUC, C121 / VIRAT / SYN pi
  P1_fig_mi_vs_rho.png      MI vs open rate rho for 4 families (G1 content-aware, G4 periodic, G5 rate-limited, G7 RR)
  P1_fig_auc_noisy.png      A1 AUC oracle vs power/traffic noisy, core gates
Run after `attacks.py --merge`, `verify_bounds.py [thm2 thm1 thm3 thm3seeds thm2v2]`, `rho_sweep.py`, `syn_v2.py`.
Also writes submission_package/numbers.tex (P2 T17).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from schedules import load_config, results_dir

CFG = load_config()
OUT = results_dir(CFG)
CORPORA = ["C121", "VIRAT", "SYN_pi0.02", "SYN_pi0.05", "SYN_pi0.1", "SYN_pi0.2", "SYN_pi0.5"]


def tau_of(p):
    return float(p.split("=")[1])


def fig_cap(mi, att):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = plt.cm.tab10(np.arange(len(CORPORA)))
    for corpus, col in zip(CORPORA, cols):
        tier = "evaluable" if corpus == "C121" else "all"
        s = mi[(mi.corpus == corpus) & (mi.family == "G1_ca_shield") & (mi.observer == "oracle") & (mi.W_s == 1)
               & (mi.target == "A1") & (mi.estimator == "plugin_corr") & (mi.tier == tier)].copy()
        if len(s):
            s["tau"] = s.param.map(tau_of)
            s = s.sort_values("tau")
            ax[0].plot(s.tau, s["mean"], "o-", color=col, ms=3, label=f"{corpus} ({tier})")
            ax[0].fill_between(s.tau, s.ci_lo, s.ci_hi, color=col, alpha=0.12)
        a = att[(att.corpus == corpus) & (att.family == "G1_ca_shield") & (att.observer == "oracle") & (att.W_s == 1)
                & (att.target == "A1") & (att.model == "gbm") & (att.tier == tier)].copy()
        if len(a):
            a["tau"] = a.param.map(tau_of)
            a = a.sort_values("tau")
            ax[1].plot(a.tau, a.value, "o-", color=col, ms=3, label=corpus)
            ax[1].fill_between(a.tau, a.ci_lo, a.ci_hi, color=col, alpha=0.12)
    ax[0].set_xlabel(r"threshold $\tau$ (higher = fewer opens)")
    ax[0].set_ylabel("I(E; N_W) bit/window, plug-in corrected")
    ax[0].set_title("A1 leakage vs tau, oracle, W = 1 s, G1 (shield)", fontsize=9)
    ax[1].set_xlabel(r"threshold $\tau$")
    ax[1].set_ylabel("A1 AUC (GBM, tau-agnostic attacker)")
    ax[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    ax[1].set_title("A1 attack AUC vs tau, oracle, W = 1 s", fontsize=9)
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "P1_fig_cap_tau.png", dpi=160)


def fig_mi_rho(mi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [("G1_ca_shield", "content-aware + shield", "#c0392b", "o-"), ("G2_ca_pure", "content-aware pure", "#e67e22", "s-"),
            ("G4_periodic", "periodic", "#2c3e50", "^-"), ("G5_rate_limited", "rate-limited K/25", "#2980b9", "D-"),
            ("G6_or_dummy", "OR-dummy (eps = inf)", "#8e44ad", "v-"), ("G7_rr", "RR-certified (Thm 3)", "#27ae60", "P")]
    panels = [("C121", "evaluable"), ("VIRAT", "all"), ("SYN_pi0.05", "all"), ("SYN_pi0.2", "all")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False)
    for ax, (corpus, tier) in zip(axes, panels):
        s = mi[(mi.corpus == corpus) & (mi.observer == "oracle") & (mi.W_s == 1) & (mi.target == "A1")
               & (mi.estimator == "plugin_corr") & (mi.tier == tier)]
        for fam, name, col, sty in fams:
            g = s[s.family == fam].sort_values("rho_mean")
            if fam == "G7_rr":
                ax.scatter(g.rho_mean, g["mean"], color=col, marker="P", s=18, label=name)
            else:
                ax.plot(g.rho_mean, g["mean"], sty, color=col, ms=3, label=name)
        d = s[s.family == "G3_deployed"]
        ax.scatter(d.rho_mean, d["mean"], marker="*", s=90, color="k", label="deployed (replay)")
        ax.set_title(f"{corpus} ({tier}), A1, W = 1 s, oracle", fontsize=8)
        ax.set_xlabel("open rate ρ")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("I(E; N_W) bit/window")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "P1_fig_mi_vs_rho.png", dpi=160)


def fig_auc_noisy(att):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("G1_ca_shield", "tau=0.55"), ("G3_deployed", "hyst"), ("G5_rate_limited", "K=5"),
            ("G6_or_dummy", "lam=0.2"), ("G7_rr", "K=5,p1=0.7,lam=0.1"), ("G7_rr", "K=inf,p1=0.5,lam=0.2"),
            ("G4_periodic", "P=5")]
    panels = [("C121", "evaluable"), ("VIRAT", "all"), ("SYN_pi0.05", "all"), ("SYN_pi0.2", "all")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (corpus, tier) in zip(axes, panels):
        s = att[(att.corpus == corpus) & (att.target == "A1") & (att.model == "gbm") & (att.tier == tier) & (att.W_s == 1)]
        xs = np.arange(len(keys))
        for j, (obs, col) in enumerate((("oracle", "#333"), ("power_head", "#d35400"), ("traffic_head", "#2471a3"))):
            v = [s[(s.family == f) & (s.param == p) & (s.observer == obs)] for f, p in keys]
            val = [x.value.iloc[0] if len(x) else np.nan for x in v]
            lo = [x.value.iloc[0] - x.ci_lo.iloc[0] if len(x) else 0 for x in v]
            hi = [x.ci_hi.iloc[0] - x.value.iloc[0] if len(x) else 0 for x in v]
            ax.errorbar(xs + (j - 1) * 0.22, val, yerr=[lo, hi], fmt="o", color=col, ms=4, label=obs, capsize=2)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.axhline(0.8, color="red", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{f.split('_')[0]}\n{p}" for f, p in keys], fontsize=5)
        ax.set_title(f"{corpus} ({tier}) A1 AUC, W = 1 s", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUC (GBM)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "P1_fig_auc_noisy.png", dpi=160)


def pick(df, **kw):
    s = df
    for k, v in kw.items():
        s = s[s[k] == v]
    return s


def numbers(mi, att, loc):
    N = {}

    def put(key, row, col="value"):
        if len(row):
            r = row.iloc[0]
            N[key] = {"value": round(float(r[col]), 4)}
            if "ci_lo" in r:
                N[key].update(ci=[round(float(r["ci_lo"]), 4), round(float(r["ci_hi"]), 4)])
            if "n_videos" in r:
                N[key]["n_videos"] = int(r["n_videos"])

    # criterion 2: A1 AUC, G1 tau = 0.55, W = 1 s, GBM, C121
    for obs in ("oracle", "power_head", "traffic_head"):
        for tier in ("all", "informative", "evaluable"):
            for fam, par in (("G1_ca_shield", "tau=0.55"), ("G3_deployed", "hyst")):
                put(f"auc_A1_C121_{fam}_{obs}_{tier}_W1",
                    pick(att, corpus="C121", family=fam, param=par, observer=obs, tier=tier, W_s=1, target="A1",
                         model="gbm"))
    # criterion 3: SYN MI A1 W = 10 s, G1 tau = 0.55, oracle, tier all + entropy ceiling
    for c in CORPORA:
        for W in (1, 10):
            s = pick(mi, corpus=c, family="G1_ca_shield", param="tau=0.55", observer="oracle", W_s=W, target="A1",
                     estimator="plugin_corr", tier="all")
            if len(s):
                put(f"mi_A1_{c}_G1_tau055_oracle_all_W{W}", s, "mean")
                N[f"HE_A1_{c}_W{W}"] = {"value": round(float(s.H_E_mean.iloc[0]), 4)}
    # deployed + RR headline MI
    for c in CORPORA:
        for fam, par in (("G3_deployed", "hyst"), ("G7_rr", "K=5,p1=0.7,lam=0.1"), ("G4_periodic", "P=5"),
                         ("G5_rate_limited", "K=5"), ("G6_or_dummy", "lam=0.2")):
            tier = "evaluable" if c == "C121" else "all"
            put(f"mi_A1_{c}_{fam}_{par}_oracle_{tier}_W1",
                pick(mi, corpus=c, family=fam, param=par, observer="oracle", W_s=1, target="A1",
                     estimator="plugin_corr", tier=tier), "mean")
    # A2 localisation headline
    for c in CORPORA:
        for est in ("cusum", "mf"):
            s = pick(loc, corpus=c, family="G1_ca_shield", param="tau=0.55", observer="oracle", estimator=est, tier="all")
            if len(s):
                N[f"loc_{c}_G1_tau055_oracle_{est}"] = {"MAE_frames": round(float(s.MAE_frames.iloc[0]), 2),
                                                         "P_within_L": round(float(s.P_within_L.iloc[0]), 4),
                                                         "n_onsets": int(s.n_onsets.iloc[0])}
    # bounds
    t2 = pd.read_csv(OUT / "P1_bounds_thm2.csv")
    cert = t2[t2.certified]
    N["thm2_certified_rows"] = int(len(cert))
    N["thm2_bound_holds_all"] = bool(cert.bound_holds.all())
    N["thm2_periodic_P5_maxI"] = round(float(t2[(t2.family == "G4_periodic") & (t2.param == "P=5")].I_meas.abs().max()), 6)
    if (OUT / "P1_bounds_thm3.csv").exists():
        t3 = pd.read_csv(OUT / "P1_bounds_thm3.csv")
        for _, r in t3[(t3.family == "G7_rr") & (t3.param == "K=5,p1=0.7,lam=0.1")].iterrows():
            N[f"thm3_{r.corpus}_rr_K5_p07_l01"] = {k: (round(float(r[k]), 4) if pd.notna(r[k]) else None)
                                                    for k in ("rho", "eps0", "eps_D5", "eps_D25", "mean_latency",
                                                              "iso_periodic_mean_latency", "miss_short",
                                                              "iso_periodic_miss_short", "I_A1_W1_all")}
    N["replay"] = replay_check()
    vi = pd.read_csv(OUT / "P1_virat_inventory.csv")
    vi = vi[vi.annotated == 1]
    N["virat"] = {"clips": int(len(vi)), "frames": int(vi.n_frames.sum()), "events_disc": int(vi.n_events_disc.sum()),
                  "pi_disc_min": float(vi.pi_disc.min()), "pi_disc_max": float(vi.pi_disc.max()),
                  "pi_disc_framewt": round(float((vi.pi_disc * vi.n_frames).sum() / vi.n_frames.sum()), 4)}
    numbers_p2(N, mi, att, loc)
    numbers_p2b(N, mi, att)
    # P3 T22b: real-plug validation keys (real.*), written by code/real_trace/attack_real.py — post hoc, PREREG §9
    rn = OUT / "real_trace" / "real_numbers.json"
    if rn.exists():
        N.update(json.loads(rn.read_text(encoding="utf-8")))
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"numbers.json: {len(N)} keys")
    write_tex(N)


# ------------------------------------------------------------------ P2 T17 additions
def replay_check() -> dict:
    """Recomputed (was hard-coded in P1): deployed-gate replay vs the logged deployed trace, threshold replay rate."""
    import gates as G
    from schedules import load_videos, processed_dir
    vids = load_videos(CFG)
    act = pd.read_csv(processed_dir(CFG) / CFG["data"]["activation_by_video"]).set_index(["dataset", "video"])
    n = sum(v.n for v in vids)
    dep = [int(G.deployed(v.score, v.valid).sum()) for v in vids]
    ref = [int(act.loc[(v.dataset, v.video), "n_opens_deployed"]) for v in vids]
    thr = sum(int(G.content_aware(v.score, v.valid, 0.55, True).sum()) for v in vids)
    return {"deployed_hysteresis_rho": round(sum(dep) / n, 4), "deployed_trace_rho": round(sum(ref) / n, 4),
            "threshold_replay_rho_tau055": round(thr / n, 4),
            "videos_exact_match": int(sum(a == b for a, b in zip(dep, ref))), "videos": len(vids), "frames": int(n),
            "pi_in_event": round(float(sum(int(v.e.sum()) for v in vids) / n), 4),
            **{f"n_{d}": sum(v.dataset == d for v in vids) for d in CFG["data"]["datasets"]}}


def numbers_p2(N, mi, att, loc):
    def put(key, row, col="value", lo="ci_lo", hi="ci_hi", n="n_videos"):
        if len(row):
            r = row.iloc[0]
            N[key] = {"value": round(float(r[col]), 4)}
            if lo in r and pd.notna(r[lo]):
                N[key]["ci"] = [round(float(r[lo]), 4), round(float(r[hi]), 4)]
            if n in r:
                N[key]["n_videos"] = int(r[n])

    A = dict(target="A1", model="gbm")
    # AUC vs window length, C121 evaluable, G1 tau 0.55
    for obs in ("oracle", "power_head", "traffic_head"):
        for W in (1, 2, 5, 10):
            put(f"auc_A1_C121_G1_{obs}_evaluable_W{W}",
                pick(att, corpus="C121", family="G1_ca_shield", param="tau=0.55", observer=obs, tier="evaluable",
                     W_s=W, **A))
    # tau sweep: MI + AUC (oracle, power), C121 evaluable, W = 1 s
    peak = (None, -1.0)
    for tau in (0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.72, 0.76, 0.78, 0.8, 0.83):
        par = f"tau={tau}"
        s = pick(mi, corpus="C121", family="G1_ca_shield", param=par, observer="oracle", W_s=1, target="A1",
                 estimator="plugin_corr", tier="evaluable")
        put(f"tau_mi_C121_eval_{tau}", s, "mean")
        if len(s):
            N[f"tau_mi_C121_eval_{tau}"]["rho"] = round(float(s.rho_mean.iloc[0]), 4)
            if float(s["mean"].iloc[0]) > peak[1]:
                peak = (tau, float(s["mean"].iloc[0]))
        for obs in ("oracle", "power_head"):
            put(f"tau_auc_C121_eval_{obs}_{tau}",
                pick(att, corpus="C121", family="G1_ca_shield", param=par, observer=obs, tier="evaluable", W_s=1, **A))
    N["tau_peak_C121_eval"] = {"tau": peak[0], "value": round(peak[1], 4)}
    # VIRAT AUC, all observers, W = 1 s
    for fam, par in (("G1_ca_shield", "tau=0.55"), ("G3_deployed", "hyst")):
        for obs in ("oracle", "power_head", "traffic_head"):
            put(f"auc_A1_VIRAT_{fam}_{obs}_all_W1",
                pick(att, corpus="VIRAT", family=fam, param=par, observer=obs, tier="all", W_s=1, **A))
    # cross-dataset
    cx = pd.read_csv(OUT / "P1_cross_dataset.csv")
    for _, r in cx[(cx.W_s == 1) & (cx.family == "G1_ca_shield")].iterrows():
        k = f"cross_{r.train}_to_{r.test}_{r.observer}_W1".replace("+", "")
        N[k] = {"value": round(float(r.AUC), 4), "ci": [round(float(r.ci_lo), 4), round(float(r.ci_hi), 4)],
                "n_videos": int(r.n_test_videos)}
    for tr, te in (("CDnet2014", "LASIESTA"), ("CDnet+LASIESTA", "BMC"), ("C121", "VIRAT")):
        g = cx[(cx.train == tr) & (cx.test == te) & (cx.W_s == 1)]
        N[f"cross_range_{tr}_to_{te}".replace("+", "")] = {"min": round(float(g.AUC.min()), 4),
                                                           "max": round(float(g.AUC.max()), 4)}
    # Thm 1(c): bound minus target entropy over all cells
    t1 = pd.read_csv(OUT / "P1_bounds_thm1.csv")
    N["thm1_cells"] = int(len(t1))
    N["thm1_min_bound_minus_H_A1"] = round(float((t1.bound - t1.H_A1).min()), 4)
    N["thm1_min_bound_minus_H_count"] = round(float(t1.bound_minus_H_count.min()), 4)
    # Thm 2 details, SYN v2, Cor 2.1(a)
    t2 = pd.read_csv(OUT / "P1_bounds_thm2.csv")
    c1 = t2[t2.certified]
    N["thm2_bound_holds_n"] = int(c1.bound_holds.sum())
    N["thm2_min_slack"] = round(float((c1.I_meas - c1.f_bound).min()), 6)
    N["thm2_max_I_certified"] = round(float(c1.I_meas.max()), 4)
    t2v = pd.read_csv(OUT / "P1b_bounds_thm2_synv2.csv")
    c2 = t2v[t2v.certified]
    N["thm2v2_certified_rows"] = int(len(c2))
    N["thm2v2_bound_holds"] = int(c2.bound_holds.sum())
    c21 = pd.read_csv(OUT / "P1_bounds_thm2_cor21a.csv")
    N["cor21a_rows"] = int(len(c21))
    N["cor21a_holds"] = int((c21.I_meas >= c21.f_bound - 1e-9).sum())
    # Thm 3 seed spread (P2 T17)
    sw = pd.read_csv(OUT / "P2_thm3_seed_sweep.csv")
    for corpus, g in sw.groupby("corpus"):
        rr = g[g.family == "G7_rr"]
        lw = [int((x.mean_latency < x.iso_periodic_mean_latency).sum()) for _, x in rr.groupby("seed")]
        mw = [int((x.miss_short < x.iso_periodic_miss_short).sum()) for _, x in rr.groupby("seed")]
        N[f"thm3seeds_{corpus}_rr_latwins"] = {"min": min(lw), "max": max(lw), "of": 24}
        N[f"thm3seeds_{corpus}_rr_shortmisswins"] = {"min": min(mw), "max": max(mw), "of": 24}
        for fam in ("G1_ca_shield", "G3_deployed"):
            f = g[g.family == fam]
            has = bool(f.miss_short.notna().any())
            N[f"thm3seeds_{corpus}_{fam}"] = {
                "lat": round(float(f.mean_latency.iloc[0]), 3),
                "per_lat_min": round(float(f.iso_periodic_mean_latency.min()), 3),
                "per_lat_max": round(float(f.iso_periodic_mean_latency.max()), 3),
                "ms": round(float(f.miss_short.iloc[0]), 3) if has else None,
                "per_ms_min": round(float(f.iso_periodic_miss_short.min()), 3) if has else None,
                "per_ms_max": round(float(f.iso_periodic_miss_short.max()), 3) if has else None,
                "rho": round(float(f.rho.mean()), 3)}
    t3 = pd.read_csv(OUT / "P1_bounds_thm3.csv")
    for _, r in t3[(t3.family == "G7_rr") & (t3.param == "K=inf,p1=0.5,lam=0.2")].iterrows():
        N[f"thm3_{r.corpus}_rr_Kinf_p05_l02"] = {k: round(float(r[k]), 4) for k in ("rho", "eps0", "eps_D5", "eps_D25")}
    # T13 rho sweep (P1b)
    rs = pd.read_csv(OUT / "P1b_rho_sweep.csv")
    ok = rs[(rs.feasible == 1) & (rs.degenerate == 0)]
    cells = wins = losses = 0
    for m in ("lat", "miss_short", "miss_long"):
        v = ok[ok[f"{m}_diff_lo"].notna()]
        cells += len(v)
        wins += int(v[f"{m}_win"].sum())
        losses += int(v[f"{m}_loss"].sum())
    N["t13"] = {"configs": int(len(rs)), "feasible": int(rs.feasible.sum()), "degenerate": int(rs.degenerate.sum()),
                "cells": cells, "wins": wins, "losses": losses, "ties": cells - wins - losses,
                "nonwins": cells - wins}
    w = ok[(ok.corpus == "C121_eval") & (ok.family == "G2_pure") & (ok.rho_target == 0.05)].iloc[0]
    N["t13_win_cell"] = {k: round(float(w[k]), 4) for k in ("miss_short_ca", "miss_short_per", "miss_short_diff",
                                                             "miss_short_diff_lo", "miss_short_diff_hi", "lat_ca",
                                                             "lat_per", "miss_long_ca", "miss_long_per")}
    for corpus, g in ok.groupby("corpus"):
        N[f"t13_{corpus}"] = {"MI_ca_min": round(float(g.MI_ca.min()), 4), "MI_ca_max": round(float(g.MI_ca.max()), 4),
                              "MI_per_max": round(float(g.MI_per.max()), 4)}
    # SYN v2 (P1b T14)
    sv = pd.read_csv(OUT / "P1b_syn_v2_mi.csv")
    for _, r in sv[sv.family == "G1_ca_shield"].iterrows():
        N[f"synv_{r.model}_pi{r.pi}_W{int(r.W_s)}"] = {"value": round(float(r.MI_corr), 4),
                                                        "ci": [round(float(r.MI_lo), 4), round(float(r.MI_hi), 4)],
                                                        "H": round(float(r.H_E), 4), "rho": round(float(r.rho), 4)}
    N["synv2_max_MI"] = round(float(sv[(sv.model == "v2") & (sv.family == "G1_ca_shield")].MI_corr.max()), 4)
    # KSG floor at true zero leakage (periodic), C121 evaluable, W = 1 s
    k = pick(mi, corpus="C121", family="G4_periodic", observer="oracle", W_s=1, target="A1", estimator="ksg",
             tier="evaluable")
    N["ksg_floor_periodic_C121_eval"] = {"min": round(float(k["mean"].min()), 4), "max": round(float(k["mean"].max()), 4)}
    # A3 occupancy R2 (appendix), C121 evaluable, G1 tau 0.55, W = 5 s
    for obs in ("oracle", "power_head"):
        put(f"a3occ_R2_C121_G1_{obs}_evaluable_W5",
            pick(att, corpus="C121", family="G1_ca_shield", param="tau=0.55", observer=obs, tier="evaluable", W_s=5,
                 target="A3occ", metric="R2_pooled"))


def numbers_p2b(N, mi, att):
    """P2 T18: remaining numbers printed in the manuscript (observer sweep, RR K = inf, protocol parameters)."""
    import attacks as AT
    sw = pd.read_csv(OUT / "P1_observer_sweep.csv")
    s = sw[(sw.corpus == "C121") & (sw.family == "G1_ca_shield") & (sw.param == "tau=0.55") & (sw.W_s == 1)
           & (sw.tier == "evaluable")]
    for _, r in s.iterrows():
        N[f"obs_{r.observer}_C121_eval_W1"] = {"value": round(float(r.AUC), 4),
                                               "ci": [round(float(r.AUC_lo), 4), round(float(r.AUC_hi), 4)],
                                               "mi": round(float(r.MI_corr), 4)}
    N["obs_sweep_C121_eval_W1_auc"] = {"min": round(float(s.AUC.min()), 4), "max": round(float(s.AUC.max()), 4)}
    for c, tier in (("C121", "evaluable"), ("VIRAT", "all")):
        for par in ("K=inf,p1=0.5,lam=0.2", "K=inf,p1=0.9,lam=0.02", "K=5,p1=0.5,lam=0.2"):
            r = pick(mi, corpus=c, family="G7_rr", param=par, observer="oracle", W_s=1, target="A1",
                     estimator="plugin_corr", tier=tier)
            if len(r):
                k = "mi_rr_" + c + "_" + par.replace("=", "").replace(",", "_")
                N[k] = {"value": round(float(r["mean"].iloc[0]), 4),
                        "ci": [round(float(r.ci_lo.iloc[0]), 4), round(float(r.ci_hi.iloc[0]), 4)],
                        "rho": round(float(r.rho_mean.iloc[0]), 4)}
    g = mi[(mi.corpus == "C121") & (mi.family == "G7_rr") & (mi.observer == "oracle") & (mi.W_s == 1)
           & (mi.target == "A1") & (mi.estimator == "plugin_corr") & (mi.tier == "evaluable")]
    k5, ki = g[g.param.str.startswith("K=5")], g[g.param.str.startswith("K=inf")]
    N["rr_grid_C121_eval"] = {"K5_min": round(float(k5["mean"].min()), 4), "K5_max": round(float(k5["mean"].max()), 4),
                              "Kinf_min": round(float(ki["mean"].min()), 4), "Kinf_max": round(float(ki["mean"].max()), 4)}
    # worked example of the Thm 2 bound (closed form; L = 5, pi = 0.1)
    import mi_estimators as Mx
    ex = {}
    for r0 in (0.02, 0.05, 0.1, 0.15):
        x = 5 * r0
        ex[f"f_r{r0}"] = round(Mx.h2(0.1 + 0.9 * x) - 0.9 * Mx.h2(x), 4)
        ex[f"fa_r{r0}"] = round(x, 4)
    ex["H_pi"] = round(Mx.h2(0.1), 4)
    N["thm2_example_L5_pi01"] = ex
    ob = CFG["observer"]
    gs = CFG["synthetic"]
    N["cfg"] = {"E_active_mJ": float(ob["E_active_mJ"]), "E_idle_mJ": float(ob["E_idle_mJ"]),
                "P_base_W": float(ob["P_base_W"]), "quant_W": float(ob["power"]["quant_W"]),
                "fs_head": float(AT.OBS_HEAD["power_head"][1]), "sigma_head": float(AT.OBS_HEAD["power_head"][2]),
                "jitter_head_ms": float(AT.OBS_HEAD["traffic_head"][1]), "n_shuffle": int(CFG["mi"]["n_shuffle"]),
                "n_boot": int(CFG["mi"]["n_boot"]), "L": int(CFG["certificate_L"]),
                "T_refresh": int(CFG["deployed_gate"]["T_refresh"]), "W_maxskip": int(CFG["deployed_gate"]["W_maxskip"]),
                "tau_dep": float(CFG["deployed_gate"]["tau"]), "open_t": float(CFG["deployed_hysteresis"]["open_t"]),
                "close_t": float(CFG["deployed_hysteresis"]["close_t"]),
                "wd_trig": int(CFG["deployed_hysteresis"]["watchdog_closed_frames"]),
                "wd_int": int(CFG["deployed_hysteresis"]["watchdog_interval"]),
                "syn_T": int(gs["T"]), "syn_videos": int(gs["n_videos"]), "syn_fps": int(gs["fps"]),
                "virat_fps": int(CFG["virat"]["fps"]), "c121_fps": int(CFG["fps"]["CDnet2014"]),
                "informative_max": float(CFG["mi"]["informative_open_rate_max"]),
                "W_per_open_1s": round(float(ob["E_active_mJ"]) * int(CFG["fps"]["CDnet2014"]) / 1000.0, 3),
                "r_active_idle": round(float(ob["E_active_mJ"]) / float(ob["E_idle_mJ"]), 1)}   # P3 B9: energy as ratio r
    m = json.loads((OUT / "P1_synthetic_model.json").read_text(encoding="utf-8"))
    m2 = json.loads((OUT / "P1b_synthetic_model_v2.json").read_text(encoding="utf-8"))
    N["synmodel"] = {"phi": round(float(m["phi"]), 3), "mean_D": round(float(m["mean_D"]), 1),
                     "median_D": float(m["median_D"]), "frac_D_lt_5": round(float(m["frac_D_lt_5"]), 3),
                     "n_events": int(m["n_events"]), "score_out_v1": round(float(m["score_mean_out"]), 3),
                     "score_in": round(float(m["score_mean_in"]), 3), "score_out_v2": round(float(m2["score_mean_out"]), 3),
                     "n_off_virat": int(m2["n_off_frames_virat"])}


def _fmt(v, nd=3):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    if isinstance(v, float):
        if abs(v) < 0.5 * 10 ** (-nd):
            v = 0.0                      # no "-0.00"
        return f"{v:.{nd}f}"
    return str(v)


def write_tex(N):
    r"""submission_package/numbers.tex: \num{key} -> value (3 dp); \num{key@2}, \num{key@1} -> 2 / 1 dp;
    nested fields key.sub; CI key.lo / key.hi. An unknown key is a LaTeX error (no silent blanks)."""
    flat = {}
    for k, v in N.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk == "ci":
                    flat[f"{k}.lo"], flat[f"{k}.hi"] = vv
                elif kk == "value":
                    flat[k] = vv
                    flat[f"{k}.value"] = vv
                elif vv is not None:
                    flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    lines = ["% AUTO-GENERATED by code/p1_report.py from results/numbers.json -- do not edit by hand.",
             r"\makeatletter",
             r"\newcommand{\num}[1]{\ifcsname nj@#1\endcsname\csname nj@#1\endcsname\else"
             r"\errmessage{numbers.tex: unknown key #1}\fi}"]
    for k, v in sorted(flat.items()):
        lines.append(r"\expandafter\def\csname nj@" + k + r"\endcsname{" + _fmt(v) + "}")
        if isinstance(v, float):
            for nd in (0, 1, 2, 3):
                lines.append(r"\expandafter\def\csname nj@" + k + "@" + str(nd) + r"\endcsname{" + _fmt(v, nd) + "}")
    lines.append(r"\makeatother")
    path = OUT.parent / "submission_package" / "numbers.tex"
    path.parent.mkdir(parents=True, exist_ok=True)       # absent in the public artifact: created on demand
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"numbers.tex: {len(flat)} keys -> {path}")


def main():
    mi = pd.read_csv(OUT / "P1_mi_table.csv")
    att = pd.read_csv(OUT / "P1_attack_table.csv")
    loc = pd.read_csv(OUT / "P1_attack_loc_table.csv")
    fig_cap(mi, att)
    fig_mi_rho(mi)
    fig_auc_noisy(att)
    numbers(mi, att, loc)


if __name__ == "__main__":
    main()
