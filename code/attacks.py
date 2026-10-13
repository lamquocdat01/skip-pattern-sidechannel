"""attacks.py — T09: full attack + MI evaluation (PREREG_39 §3–§7).

Parts (run per corpus; outputs in results/p1_parts/, merged by --merge):
  mi     plug-in (+perm null) and KSG MI, oracle, FULL gate grid; + noisy headline observers on the core set
  attack A1 / A2-window classifiers (GBM main, logistic), A3 regression (HGB, ridge), GroupKFold-5 by video,
         tau-agnostic attacker for G1/G3; core set x {oracle, power_head, traffic_head} x W in {1,2,5,10} s;
         classifier lower bound on MI from out-of-fold GBM probabilities
  loc    A2 onset localisation: CUSUM (trained per fold) and matched filter; MAE (frames), Pr[|err| <= L]
  sweep  observer sweeps (power fs x sigma, traffic jitter) for the sweep set, W in {1,5} s: A1 GBM AUC + plug-in MI
  cross  cross-dataset A1 (CDnet -> LASIESTA, CDnet+LASIESTA -> BMC, C121 -> VIRAT), G1 tau = 0.55

Usage:  python attacks.py --corpus C121 --parts mi attack loc sweep
        python attacks.py --cross
        python attacks.py --merge
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 4 single-thread queues beat one multi-thread run (timed 09-10)
import psutil  # noqa: E402

import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gates as G
import mi_estimators as M
import observer as obs
from data_sparse import load_corpus
from schedules import load_config, results_dir

CFG = load_config()
OCFG = CFG["observer"]
L_CERT = CFG["certificate_L"]
W_S = CFG["mi"]["windows_s"]
N_SHUFFLE = CFG["mi"]["n_shuffle"]
N_BOOT = CFG["mi"]["n_boot"]
TRAIN_CAP = 20000
OUT = results_dir(CFG) / "p1_parts"
OUT.mkdir(parents=True, exist_ok=True)

OBS_HEAD = {"oracle": ("oracle",), "power_head": ("power", 1.0, 0.05), "traffic_head": ("traffic", 50.0)}
OBS_SWEEP = {**{f"power_fs{fs}_s{s}": ("power", float(fs), s) for fs in (1, 2, 5, 10) for s in (0.0, 0.02, 0.05, 0.10)},
             **{f"traffic_j{j}": ("traffic", float(j)) for j in (5, 50, 200)}}


def seed(*parts) -> np.random.Generator:
    return np.random.default_rng([42] + [zlib.crc32(str(p).encode()) for p in parts])


# ------------------------------------------------------------------ schedules / observations
def schedules(clips, grid, corpus):
    """dict (family, label) -> list of a (one per clip)."""
    out = {}
    for fam, lab, p in grid:
        out[(fam, lab)] = [G.make(c.score, c.valid, fam, p, seed(corpus, fam, lab, c.key), L_CERT) for c in clips]
    return out


def observe(a: np.ndarray, fps: float, spec: tuple, rng) -> tuple[np.ndarray, float]:
    """Per-frame estimated open count x_t and the binarisation threshold for gap/run features."""
    if spec[0] == "oracle":
        return a.astype(np.float64), 0.5
    if spec[0] == "power":
        fs, sigma = spec[1], spec[2]
        pw = obs.power_trace(a, fps, OCFG, fs, sigma, rng)
        cnt = np.clip(obs.power_to_open_count(pw, fps, OCFG, fs), 0, None)
        fpi = max(1, int(round(fps / fs)))
        x = np.zeros(len(a))
        x[: len(cnt) * fpi] = np.repeat(cnt / fpi, fpi)
        return x, 0.5 / fpi
    if spec[0] == "traffic":
        t_open = np.flatnonzero(a) / fps + rng.normal(0.0, spec[1] * 1e-3, int(a.sum()))
        idx = np.clip(np.floor(t_open * fps).astype(int), 0, len(a) - 1)
        return np.bincount(idx, minlength=len(a)).astype(np.float64), 0.5
    raise ValueError(spec)


@njit(cache=True)
def _gap_feats(b):
    m, w = b.shape
    out = np.zeros((m, 4))
    for i in range(m):
        longest = 0
        cur = 0
        runs = 0
        first = -1
        last = -1
        prev = 0
        for j in range(w):
            if b[i, j]:
                if first < 0:
                    first = j
                last = j
                if prev == 0:
                    runs += 1
                if cur > longest:
                    longest = cur
                cur = 0
                prev = 1
            else:
                cur += 1
                prev = 0
        if cur > longest:
            longest = cur
        out[i, 0] = longest / w
        out[i, 1] = runs / w
        out[i, 2] = (first if first >= 0 else w) / w
        out[i, 3] = (last if last >= 0 else -1) / w
    return out


def window_feats(x: np.ndarray, w: int, thr: float) -> np.ndarray:
    m = len(x) // w
    X = x[: m * w].reshape(m, w)
    rate = X.sum(1) / w
    sb = w // 5
    sub = X[:, : 5 * sb].reshape(m, 5, sb).sum(2) / sb
    gf = _gap_feats((X >= thr).astype(np.int8))
    prev = np.r_[rate[:1], rate[:-1]]
    nxt = np.r_[rate[1:], rate[-1:]]
    return np.column_stack([rate, sub, gf, prev, nxt])


def window_targets(e: np.ndarray, w: int, fps: float) -> dict:
    m = len(e) // w
    E = e[: m * w].reshape(m, w)
    on = np.zeros(len(e), np.int8)
    on[1:] = (e[1:] == 1) & (e[:-1] == 0)
    on[0] = e[0]
    ON = on[: m * w].reshape(m, w).sum(1)
    return dict(A1=E.max(1).astype(np.int8), A2w=(ON > 0).astype(np.int8), occ=E.mean(1),
                rate=ON * 60.0 / (w / fps))


def wframes(c, W_s) -> int:
    return int(round(W_s * c.fps))


# ------------------------------------------------------------------ tiers
def deployed_rate(c) -> float:
    return float(G.deployed(c.score, c.valid).mean())


def tier_masks(clips) -> dict:
    """clip-level informative flag + per (clip, W, target) evaluable flag."""
    info = np.array([deployed_rate(c) < CFG["mi"]["informative_open_rate_max"] for c in clips])
    ev = {}
    for W_s in W_S:
        for i, c in enumerate(clips):
            t = window_targets(c.e, wframes(c, W_s), c.fps)
            nw = len(t["A1"])
            for k in ("A1", "A2w"):
                ev[(i, W_s, k)] = M.h2(float(t[k].mean())) >= 0.5 and nw >= 30
            ev[(i, W_s, "A3")] = float(np.std(t["occ"])) >= 0.1 and nw >= 30
    return dict(info=info, ev=ev)


def in_tier(tiers, i, W_s, target, tier) -> bool:
    if tier == "all":
        return True
    if tier == "informative":
        return bool(tiers["info"][i])
    return bool(tiers["ev"][(i, W_s, "A3" if target.startswith("A3") else target)])


TIERS = ("all", "informative", "evaluable")


# ------------------------------------------------------------------ part: MI
def part_mi(corpus, clips, tiers, sched):
    recs = []
    for (fam, lab), alist in sched.items():
        views = {"oracle": OBS_HEAD["oracle"]}
        if (fam, lab) in G.CORE:
            views = OBS_HEAD
        for i, (c, a) in enumerate(zip(clips, alist)):
            for oname, spec in views.items():
                x, _ = observe(a, c.fps, spec, seed(corpus, fam, lab, c.key, oname))
                for W_s in W_S:
                    w = wframes(c, W_s)
                    m = c.n // w
                    rate = x[: m * w].reshape(m, w).sum(1) / w
                    rate = np.clip(rate, 0, 1)
                    tg = window_targets(c.e, w, c.fps)
                    rng = seed(corpus, fam, lab, c.key, oname, W_s)
                    for tname in ("A1", "A2w"):
                        e = tg[tname]
                        mi, null, corr, p = M.plugin(e, rate, rng, N_SHUFFLE)
                        recs.append(dict(corpus=corpus, dataset=c.dataset, video=c.video, clip=i, family=fam,
                                         param=lab, observer=oname, W_s=W_s, target=tname, n_windows=len(e),
                                         rho=float(a.mean()), p_event=float(e.mean()), H_E=M.h2(float(e.mean())),
                                         MI_plugin=mi, MI_null=null, MI_corr=corr, p_value=p,
                                         MI_ksg=M.ksg(e, rate, rng) if len(e) >= 30 else np.nan,
                                         informative=int(tiers["info"][i]),
                                         evaluable=int(tiers["ev"][(i, W_s, tname)])))
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / f"mi_{corpus}.parquet", index=False)
    return df


# ------------------------------------------------------------------ part: attack (classifiers)
def gbm_clf():
    return HistGradientBoostingClassifier(max_iter=200, random_state=42)


def logit_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))


def build_matrix(corpus, clips, sched, keys, spec_name, W_s):
    """For every gate config in keys: X (rows), clip id per row, targets."""
    spec = {**OBS_HEAD, **OBS_SWEEP}[spec_name]
    out = {}
    for k in keys:
        Xs, ids, T = [], [], {t: [] for t in ("A1", "A2w", "occ", "rate")}
        for i, (c, a) in enumerate(zip(clips, sched[k])):
            x, thr = observe(a, c.fps, spec, seed(corpus, *k, c.key, spec_name))
            w = wframes(c, W_s)
            f = window_feats(x, w, thr)
            tg = window_targets(c.e, w, c.fps)
            Xs.append(f); ids.append(np.full(len(f), i))
            for t in T:
                T[t].append(tg[t])
        out[k] = (np.vstack(Xs), np.concatenate(ids), {t: np.concatenate(v) for t, v in T.items()})
    return out


def train_keys(k, keys_all):
    fam, lab = k
    if fam == "G1_ca_shield":
        return [kk for kk in keys_all if kk[0] == "G1_ca_shield" and kk != k]
    if fam == "G3_deployed":
        return [kk for kk in keys_all if kk[0] == "G1_ca_shield" and kk[1] != "tau=0.55"]
    return [k]


def folds(n_clips):
    n_splits = min(5, n_clips)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ids = np.arange(n_clips)
    return [(ids[tr], ids[te]) for tr, te in gkf.split(ids, groups=ids)]


def subsample(n, cap, rng):
    return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))


def run_classifiers(mats, test_keys, n_clips, models=("gbm", "logit"), targets=("A1", "A2w", "A3")):
    """Out-of-fold predictions for every test key. Returns dict key -> dict of arrays."""
    keys_all = list(mats.keys())
    res = {}
    fl = folds(n_clips)
    for k in test_keys:
        X, ids, T = mats[k]
        pred = {f"{t}_{m}": np.full(len(X), np.nan) for t in targets if t != "A3" for m in models}
        if "A3" in targets:
            pred.update({f"occ_{m}": np.full(len(X), np.nan) for m in ("gbm", "ridge")})
            pred.update({f"rate_{m}": np.full(len(X), np.nan) for m in ("gbm", "ridge")})
        tk = train_keys(k, keys_all)
        for fi, (tr, te) in enumerate(fl):
            Xtr = np.vstack([mats[kk][0][np.isin(mats[kk][1], tr)] for kk in tk])
            Ttr = {t: np.concatenate([mats[kk][2][t][np.isin(mats[kk][1], tr)] for kk in tk]) for t in T}
            sel = subsample(len(Xtr), TRAIN_CAP, seed("sub", *k, fi))
            Xtr = Xtr[sel]
            Ttr = {t: v[sel] for t, v in Ttr.items()}
            mte = np.isin(ids, te)
            if not mte.any():
                continue
            for t in [t for t in targets if t != "A3"]:
                y = Ttr[t]
                for m in models:
                    if y.min() == y.max():
                        pred[f"{t}_{m}"][mte] = float(y[0])
                        continue
                    clf = gbm_clf() if m == "gbm" else logit_clf()
                    clf.fit(Xtr, y)
                    pred[f"{t}_{m}"][mte] = clf.predict_proba(X[mte])[:, 1]
            if "A3" in targets:
                for t in ("occ", "rate"):
                    for m in ("gbm", "ridge"):
                        reg = HistGradientBoostingRegressor(max_iter=200, random_state=42) if m == "gbm" \
                            else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                        reg.fit(Xtr, Ttr[t])
                        pred[f"{t}_{m}"][mte] = reg.predict(X[mte])
        res[k] = pred
    return res


def auc_boot(y, s, cid, rng, n_boot=N_BOOT, nb=200):
    """Pooled AUC (exact) + cluster-bootstrap CI over clips (binned scores)."""
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, s)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 1)
    uc, ci = np.unique(cid, return_inverse=True)
    nbk = len(edges)
    hp = np.zeros((len(uc), nbk)); hn = np.zeros((len(uc), nbk))
    np.add.at(hp, (ci[y == 1], b[y == 1]), 1)
    np.add.at(hn, (ci[y == 0], b[y == 0]), 1)
    wts = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=n_boot).astype(float)
    Hp, Hn = wts @ hp, wts @ hn
    below = np.cumsum(Hn, 1) - Hn
    num = (Hp * (below + 0.5 * Hn)).sum(1)
    den = Hp.sum(1) * Hn.sum(1)
    a = num[den > 0] / den[den > 0]
    return auc, float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def mean_boot(v, rng, n_boot=N_BOOT):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(v, (n_boot, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarise_preds(corpus, clips, tiers, mats, res, oname, W_s, rows, mi_rows):
    for k, pred in res.items():
        X, ids, T = mats[k]
        for tier in TIERS:
            for tname in ("A1", "A2w"):
                keep = np.array([in_tier(tiers, i, W_s, tname, tier) for i in range(len(clips))])
                m = keep[ids]
                if m.sum() == 0:
                    continue
                for mdl in ("gbm", "logit"):
                    p = pred[f"{tname}_{mdl}"][m]
                    ok = ~np.isnan(p)
                    y = T[tname][m][ok]
                    if len(y) == 0:
                        continue
                    auc, lo, hi = auc_boot(y, p[ok], ids[m][ok], seed("auc", corpus, *k, oname, W_s, tier, tname, mdl))
                    rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, W_s=W_s, target=tname,
                                     model=mdl, tier=tier, metric="AUC", value=auc, ci_lo=lo, ci_hi=hi,
                                     n_videos=int(len(np.unique(ids[m]))), n_rows=int(len(y))))
                # classifier lower bound on MI, per video, GBM
                lbs = []
                for i in np.unique(ids[m]):
                    mm = (ids == i) & ~np.isnan(pred[f"{tname}_gbm"])
                    lbs.append(M.classifier_lb(T[tname][mm], pred[f"{tname}_gbm"][mm]))
                mu, lo, hi = mean_boot(lbs, seed("lb", corpus, *k, oname, W_s, tier, tname))
                mi_rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, W_s=W_s, target=tname,
                                    tier=tier, estimator="classifier_lb", mean=mu, ci_lo=lo, ci_hi=hi,
                                    n_videos=len(lbs)))
            keep = np.array([in_tier(tiers, i, W_s, "A3", tier) for i in range(len(clips))])
            m = keep[ids]
            if m.sum() == 0:
                continue
            for mdl in ("gbm", "ridge"):
                yo, po = T["occ"][m], pred[f"occ_{mdl}"][m]
                yr, pr = T["rate"][m], pred[f"rate_{mdl}"][m]
                ok = ~np.isnan(po)
                per_r2 = []
                for i in np.unique(ids[m]):
                    mm = (ids[m] == i) & ok
                    if mm.sum() > 1 and np.var(yo[mm]) > 0:
                        per_r2.append(r2_score(yo[mm], po[mm]))
                _, lo, hi = mean_boot(per_r2, seed("r2", corpus, *k, oname, W_s, tier, mdl))
                rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, W_s=W_s, target="A3occ",
                                 model=mdl, tier=tier, metric="R2_pooled", value=r2_score(yo[ok], po[ok]),
                                 ci_lo=lo, ci_hi=hi, n_videos=int(len(np.unique(ids[m]))), n_rows=int(ok.sum()),
                                 note="CI = bootstrap of per-video R2"))
                mae_v = [np.mean(np.abs(yr[(ids[m] == i) & ok] - pr[(ids[m] == i) & ok])) for i in np.unique(ids[m])]
                mu, lo, hi = mean_boot(mae_v, seed("mae", corpus, *k, oname, W_s, tier, mdl))
                rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, W_s=W_s, target="A3rate",
                                 model=mdl, tier=tier, metric="MAE_events_per_min", value=mu, ci_lo=lo, ci_hi=hi,
                                 n_videos=len(mae_v), n_rows=int(ok.sum())))


def part_attack(corpus, clips, tiers, sched):
    rows, mi_rows = [], []
    keys = [k for k in sched if k in G.CORE]
    for oname in OBS_HEAD:
        for W_s in W_S:
            ram_guard()
            t0 = time.time()
            mats = build_matrix(corpus, clips, sched, keys, oname, W_s)
            res = run_classifiers(mats, keys, len(clips))
            summarise_preds(corpus, clips, tiers, mats, res, oname, W_s, rows, mi_rows)
            print(f"  attack {corpus} {oname} W={W_s}s: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"attack_{corpus}.csv", index=False)
    pd.DataFrame(mi_rows).to_csv(OUT / f"milb_{corpus}.csv", index=False)


# ------------------------------------------------------------------ part: A2 localisation
def isolated_onsets(e: np.ndarray, quiet: int) -> np.ndarray:
    on = np.flatnonzero((e[1:] == 1) & (e[:-1] == 0)) + 1
    cs = np.r_[0, np.cumsum(e)]
    return np.array([t for t in on if t >= quiet and cs[t] - cs[t - quiet] == 0], int)


@njit(cache=True)
def _cusum(x, a, b, h):
    """Poisson-LLR CUSUM; returns (alarm times, estimated change points)."""
    S = 0.0
    last0 = 0
    al, cp = [], []
    for t in range(x.shape[0]):
        S = S + x[t] * a - b
        if S <= 0:
            S = 0.0
            last0 = t + 1
        elif S > h:
            al.append(t)
            cp.append(last0)
            S = 0.0
            last0 = t + 1
    return np.array(al, np.int64), np.array(cp, np.int64)


def cusum_params(xs, es, fps):
    lo = np.concatenate([x[e == 0] for x, e in zip(xs, es)])
    hi = []
    for x, e in zip(xs, es):
        for t in isolated_onsets(e, int(fps)):
            seg = e[t:t + int(fps)]
            d = int(seg.argmin()) if (seg == 0).any() else len(seg)
            hi.append(x[t:t + d])
    r_lo = max(float(lo.mean()) if len(lo) else 1e-4, 1e-4)
    r_hi = float(np.concatenate(hi).mean()) if hi else r_lo
    return r_lo, r_hi


def false_alarm_rate(xs, es, fps, a, b, h):
    """Alarms per minute raised in a quiet stretch (no in-event frame in the 2 s up to and including the alarm)."""
    fa, minutes = 0, 0.0
    q = int(2 * fps)
    for x, e in zip(xs, es):
        al, _ = _cusum(x, a, b, h)
        cs = np.r_[0, np.cumsum(e)]
        lo = np.maximum(al - q, 0)
        fa += int(((cs[al + 1] - cs[lo]) == 0).sum())
        minutes += len(x) / fps / 60
    return fa / max(minutes, 1e-9)


def part_loc(corpus, clips, tiers, sched):
    rows = []
    keys = [k for k in sched if k in G.CORE]
    fl = folds(len(clips))
    for k in keys:
        for oname, spec in OBS_HEAD.items():
            xs = [observe(a, c.fps, spec, seed(corpus, *k, c.key, oname, "loc"))[0] for c, a in zip(clips, sched[k])]
            errs = {"cusum": {}, "mf": {}}
            for tr, te in fl:
                fps = clips[tr[0]].fps
                r_lo, r_hi = cusum_params([xs[i] for i in tr], [clips[i].e for i in tr], fps)
                if r_hi <= r_lo * 1.0001:
                    a_, b_, h_ = 0.0, 1.0, np.inf
                else:
                    a_, b_ = np.log(r_hi / r_lo), r_hi - r_lo
                    h_ = np.inf
                    for h in np.geomspace(0.5, 500, 60):
                        if false_alarm_rate([xs[i] for i in tr], [clips[i].e for i in tr], fps, a_, b_, h) <= 1.0:
                            h_ = h
                            break
                for i in te:
                    c, x = clips[i], xs[i]
                    q = int(c.fps)
                    cap = 2 * q
                    ons = isolated_onsets(c.e, q)
                    al, cp = _cusum(x, a_, b_, h_) if np.isfinite(h_) else (np.zeros(0, int), np.zeros(0, int))
                    cs = np.r_[0.0, np.cumsum(x)]
                    ec, em = [], []
                    for t0 in ons:
                        j = np.flatnonzero((al >= t0 - q) & (al <= t0 + cap))
                        ec.append(min(abs(int(cp[j[0]]) - t0), cap) if len(j) else cap)
                        lo_, hi_ = max(q, t0 - cap), min(c.n - q, t0 + cap)
                        if hi_ <= lo_:
                            em.append(cap)
                            continue
                        tt = np.arange(lo_, hi_ + 1)
                        mfs = (cs[tt + q] - cs[tt]) - (cs[tt] - cs[tt - q])
                        em.append(min(abs(int(tt[np.argmax(mfs)]) - t0), cap))
                    errs["cusum"][i] = np.array(ec, float)
                    errs["mf"][i] = np.array(em, float)
            for est, d in errs.items():
                for tier in TIERS:
                    sel = [i for i in d if len(d[i]) and (tier == "all" or (tier == "informative" and tiers["info"][i])
                                                          or (tier == "evaluable" and len(d[i]) >= 3))]
                    if not sel:
                        continue
                    allerr = np.concatenate([d[i] for i in sel])
                    mae_v = [d[i].mean() for i in sel]
                    hit_v = [(d[i] <= L_CERT).mean() for i in sel]
                    _, lo, hi = mean_boot(mae_v, seed("loc", corpus, *k, oname, est, tier))
                    _, lo2, hi2 = mean_boot(hit_v, seed("loc2", corpus, *k, oname, est, tier))
                    rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, estimator=est, tier=tier,
                                     n_videos=len(sel), n_onsets=len(allerr), MAE_frames=allerr.mean(),
                                     MAE_ci_lo=lo, MAE_ci_hi=hi, P_within_L=(allerr <= L_CERT).mean(),
                                     P_ci_lo=lo2, P_ci_hi=hi2, cap_frames=2 * int(clips[sel[0]].fps)))
        print(f"  loc {corpus} {k}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"loc_{corpus}.csv", index=False)


# ------------------------------------------------------------------ part: observer sweeps
def part_sweep(corpus, clips, tiers, sched):
    rows = []
    g1 = [k for k in sched if k[0] == "G1_ca_shield"]
    for oname in OBS_SWEEP:
        for W_s in (1, 5):
            t0 = time.time()
            keys = sorted(set(g1) | {k for k in sched if k in G.SWEEP})
            mats = build_matrix(corpus, clips, sched, keys, oname, W_s)
            test = [k for k in keys if k in G.SWEEP]
            res = run_classifiers(mats, test, len(clips), models=("gbm",), targets=("A1",))
            for k in test:
                X, ids, T = mats[k]
                for tier in TIERS:
                    keep = np.array([in_tier(tiers, i, W_s, "A1", tier) for i in range(len(clips))])
                    m = keep[ids]
                    if m.sum() == 0:
                        continue
                    p = res[k]["A1_gbm"][m]
                    ok = ~np.isnan(p)
                    auc, lo, hi = auc_boot(T["A1"][m][ok], p[ok], ids[m][ok], seed("sw", corpus, *k, oname, W_s, tier))
                    mis = []
                    for i in np.unique(ids[m]):
                        mm = ids == i
                        mis.append(M.plugin(T["A1"][mm], np.clip(X[mm, 0], 0, 1), seed("swmi", corpus, *k, oname, W_s, i),
                                            N_SHUFFLE)[2])
                    mu, mlo, mhi = mean_boot(mis, seed("swb", corpus, *k, oname, W_s, tier))
                    rows.append(dict(corpus=corpus, family=k[0], param=k[1], observer=oname, W_s=W_s, tier=tier,
                                     AUC=auc, AUC_lo=lo, AUC_hi=hi, MI_corr=mu, MI_lo=mlo, MI_hi=mhi,
                                     n_videos=int(len(np.unique(ids[m])))))
            print(f"  sweep {corpus} {oname} W={W_s}: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"sweep_{corpus}.csv", index=False)


# ------------------------------------------------------------------ cross-dataset
def part_cross():
    c121 = load_corpus("C121")
    virat = load_corpus("VIRAT")
    rows = []
    chains = [("CDnet2014", ["CDnet2014"], "LASIESTA"), ("CDnet+LASIESTA", ["CDnet2014", "LASIESTA"], "BMC"),
              ("C121", ["CDnet2014", "LASIESTA", "BMC"], "VIRAT")]
    k = ("G1_ca_shield", "tau=0.55")
    for oname in OBS_HEAD:
        for W_s in W_S:
            for name, tr_sets, te_set in chains:
                tr = [c for c in c121 if c.dataset in tr_sets]
                te = virat if te_set == "VIRAT" else [c for c in c121 if c.dataset == te_set]
                Xs = {}
                for part, cl in (("tr", tr), ("te", te)):
                    sched = {kk: [G.make(c.score, c.valid, kk[0], dict(tau=float(kk[1][4:])),
                                         seed("cross", *kk, c.key), L_CERT) for c in cl]
                             for kk in ([kk for kk in [("G1_ca_shield", f"tau={t}") for t in G.TAUS] if kk != k]
                                        if part == "tr" else [k])}
                    Xs[part] = build_matrix("cross", cl, sched, list(sched), oname, W_s)
                Xtr = np.vstack([v[0] for v in Xs["tr"].values()])
                ytr = np.concatenate([v[2]["A1"] for v in Xs["tr"].values()])
                sel = subsample(len(Xtr), TRAIN_CAP, seed("crosssub", name, oname, W_s))
                clf = gbm_clf().fit(Xtr[sel], ytr[sel])
                X, ids, T = Xs["te"][k]
                p = clf.predict_proba(X)[:, 1]
                auc, lo, hi = auc_boot(T["A1"], p, ids, seed("crossauc", name, oname, W_s))
                rows.append(dict(train=name, test=te_set, observer=oname, W_s=W_s, family=k[0], param=k[1],
                                 AUC=auc, ci_lo=lo, ci_hi=hi, n_test_videos=len(te), n_rows=len(p)))
                print(f"  cross {name}->{te_set} {oname} W={W_s}: AUC {auc:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cross_dataset.csv", index=False)


# ------------------------------------------------------------------ merge
def merge():
    out = results_dir(CFG)
    att = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("attack_*.csv"))], ignore_index=True)
    att.to_csv(out / "P1_attack_table.csv", index=False)
    loc = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("loc_*.csv"))], ignore_index=True)
    loc.to_csv(out / "P1_attack_loc_table.csv", index=False)
    sw = [pd.read_csv(f) for f in sorted(OUT.glob("sweep_*.csv"))]
    if sw:
        pd.concat(sw, ignore_index=True).to_csv(out / "P1_observer_sweep.csv", index=False)
    if (OUT / "cross_dataset.csv").exists():
        pd.read_csv(OUT / "cross_dataset.csv").to_csv(out / "P1_cross_dataset.csv", index=False)
    mi = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("mi_*.parquet"))], ignore_index=True)
    rows = []
    keys = ["corpus", "family", "param", "observer", "W_s", "target"]
    for tier in TIERS:
        sub = mi if tier == "all" else mi[mi[tier] == 1]
        for k, g in sub.groupby(keys):
            rng = seed("mi", *k, tier)
            for est, col in (("plugin_corr", "MI_corr"), ("ksg", "MI_ksg")):
                mu, lo, hi = mean_boot(g[col].to_numpy(), rng)
                rows.append(dict(zip(keys, k), tier=tier, estimator=est, mean=mu, ci_lo=lo, ci_hi=hi,
                                 n_videos=len(g), rho_mean=g.rho.mean(), H_E_mean=g.H_E.mean(),
                                 frac_p_lt_0_05=float((g.p_value < 0.05).mean()) if est == "plugin_corr" else np.nan))
    lb = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("milb_*.csv"))], ignore_index=True)
    tab = pd.concat([pd.DataFrame(rows), lb], ignore_index=True)
    tab.to_csv(out / "P1_mi_table.csv", index=False)
    print("merged:", len(att), "attack rows,", len(tab), "MI rows")


def ram_guard():
    free = psutil.virtual_memory().available / 2**30
    if free < CFG["virat"]["min_free_ram_gb"]:
        sys.exit(f"free RAM {free:.1f} GB < {CFG['virat']['min_free_ram_gb']} GB — stopping (rerun resumes)")


PART_FILE = {"mi": "mi_{}.parquet", "attack": "attack_{}.csv", "loc": "loc_{}.csv", "sweep": "sweep_{}.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--parts", nargs="*", default=["mi", "attack", "loc", "sweep"])
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    for corpus in a.corpus:
        t0 = time.time()
        clips = load_corpus(corpus)
        tiers = tier_masks(clips)
        grid = G.full_grid(include_g8=corpus.startswith("SYN"))
        sched = schedules(clips, grid, corpus)
        print(f"{corpus}: {len(clips)} clips, {sum(c.n for c in clips)} frames, "
              f"informative {int(tiers['info'].sum())}, schedules {time.time() - t0:.0f}s", flush=True)
        for part in a.parts:
            if (OUT / PART_FILE[part].format(corpus)).exists():
                print(f"{corpus} {part}: exists, skipped", flush=True)
                continue
            ram_guard()
            t1 = time.time()
            {"mi": part_mi, "attack": part_attack, "loc": part_loc, "sweep": part_sweep}[part](corpus, clips, tiers, sched)
            print(f"{corpus} {part}: {time.time() - t1:.0f}s", flush=True)
    if a.cross:
        part_cross()
    if a.merge:
        merge()


if __name__ == "__main__":
    main()
