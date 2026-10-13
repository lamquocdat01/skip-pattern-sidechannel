"""gates.py — P1 gate families G1–G8 (PREREG_39 §5), numba-compiled.

Every gate maps (score S_t, valid_t) [+ private randomness] to an open schedule a_t in {0,1}.
G1 content_aware_shield  demand S_t > tau OR forced refresh every T_refresh OR reuse cap (deployed `fsl > W`, L = W+2)
G2 content_aware_pure    S_t > tau
G3 deployed              exact replay of the deployed ASMAG+ trace: hysteresis (open >= 0.55, close <= 0.30),
                         watchdog (closed >= 12 frames -> open every 6), refresh 15, reuse cap. Matches
                         activation_by_video.n_opens_deployed on 121/121 videos (checked 09-10-2026).
G4 periodic              (t - phase) mod P == 0, phase ~ Unif independent of content
G5 rate_limited          sliding-window bucket (<= K content opens in the last Wb frames) + oblivious floor P = 5
G6 or_dummy              G5 OR Bernoulli(lam) dummies (P0 "privacy_certified"; p1 = 1 -> epsilon = inf)
G7 rr_certified          Thm 3: floor P = L OR randomized response on TUMBLING-bucket demand (first K demands per
                         aligned Wb-window, computed on demand before RR) with Pr[open] = p1 if admitted else lam
G8 ca_refresh            S_t > tau OR refresh every P_r (no reuse cap) — only for the Thm 2 check on SYN
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF_K = 10**9


@njit(cache=True)
def _shield(demand, valid, T_refresh, W_maxskip, fixed):
    n = demand.shape[0]
    a = np.zeros(n, np.int8)
    fsl = W_maxskip + 1
    for t in range(n):
        if valid[t] == 0:
            fsl += 1
            continue
        cap = fsl >= W_maxskip if fixed else fsl > W_maxskip
        o = demand[t] or (t > 0 and t % T_refresh == 0) or cap
        if o:
            a[t] = 1
            fsl = 0
        else:
            fsl += 1
    return a


@njit(cache=True)
def _deployed(score, valid, open_t, close_t, wd_trig, wd_int, T_refresh, W_maxskip):
    n = score.shape[0]
    a = np.zeros(n, np.int8)
    g = False
    cc = 0
    fsl = W_maxskip + 1
    for t in range(n):
        if valid[t] == 0:
            fsl += 1
            continue
        s = score[t]
        if s >= open_t:
            g = True
        elif s <= close_t:
            g = False
        wd = False
        if not g:
            cc += 1
            if cc >= wd_trig and cc % wd_int == 0:
                wd = True
        else:
            cc = 0
        o = g or wd or (t > 0 and t % T_refresh == 0) or fsl > W_maxskip
        if o:
            a[t] = 1
            fsl = 0
        else:
            fsl += 1
    return a


@njit(cache=True)
def _sliding_bucket(demand, floor, K, Wb):
    n = demand.shape[0]
    c = np.zeros(n, np.int8)
    run = 0
    for t in range(n):
        if t >= Wb:
            run -= c[t - Wb]
        if demand[t] and floor[t] == 0 and run < K:
            c[t] = 1
            run += 1
    return c


@njit(cache=True)
def _tumbling_bucket(demand, K, Wb):
    n = demand.shape[0]
    c = np.zeros(n, np.int8)
    used = 0
    for t in range(n):
        if t % Wb == 0:
            used = 0
        if demand[t] and used < K:
            c[t] = 1
            used += 1
    return c


def periodic(n: int, P: int, phase: int = 0) -> np.ndarray:
    a = np.zeros(n, np.int8)
    a[phase % P::P] = 1
    return a


def demand_of(score, valid, tau):
    return ((score > tau) & (valid == 1)).astype(np.int8)


def content_aware(score, valid, tau, shield=True, T_refresh=15, W_maxskip=4, reuse_logic="deployed"):
    d = demand_of(score, valid, tau)
    if not shield:
        return d
    return _shield(d, valid.astype(np.int8), T_refresh, W_maxskip, reuse_logic == "fixed")


def deployed(score, valid, open_t=0.55, close_t=0.30, wd_trig=12, wd_int=6, T_refresh=15, W_maxskip=4):
    return _deployed(score.astype(np.float64), valid.astype(np.int8), open_t, close_t, wd_trig, wd_int,
                     T_refresh, W_maxskip)


def rate_limited(score, valid, tau, K, Wb, floor_P=5, rng=None):
    n = len(score)
    phase = 0 if rng is None else int(rng.integers(floor_P))
    floor = periodic(n, floor_P, phase)
    c = _sliding_bucket(demand_of(score, valid, tau), floor, K, Wb)
    return np.maximum(c, floor)


def or_dummy(score, valid, tau, K, Wb, lam, rng, floor_P=5):
    a = rate_limited(score, valid, tau, K, Wb, floor_P, rng)
    return np.maximum(a, (rng.random(len(score)) < lam).astype(np.int8))


def rr_certified(score, valid, tau, K, Wb, p1, lam, rng, L=5, return_parts=False):
    n = len(score)
    floor = periodic(n, L, int(rng.integers(L)))
    c = _tumbling_bucket(demand_of(score, valid, tau), min(K, INF_K), Wb)
    u = rng.random(n)
    r = np.where(c == 1, u < p1, u < lam).astype(np.int8)
    a = np.maximum(floor, r)
    return (a, floor, c) if return_parts else a


def ca_refresh(score, valid, tau, P_r, rng):
    n = len(score)
    return np.maximum(demand_of(score, valid, tau), periodic(n, P_r, int(rng.integers(P_r))))


# ------------------------------------------------------------------ PREREG grid
TAUS = [0.05, 0.20, 0.35, 0.45, 0.55, 0.65, 0.72, 0.76, 0.78, 0.80, 0.83]
G7_GRID = [(K, p1, lam) for K in (5, INF_K) for p1 in (0.5, 0.7, 0.9) for lam in (0.02, 0.05, 0.1, 0.2)]


def kname(K):
    return "inf" if K >= INF_K else str(K)


def full_grid(include_g8: bool = False):
    """(family, label, params) for every PREREG §5 configuration."""
    g = []
    for tau in TAUS:
        g.append(("G1_ca_shield", f"tau={tau}", dict(tau=tau)))
    for tau in TAUS:
        g.append(("G2_ca_pure", f"tau={tau}", dict(tau=tau)))
    g.append(("G3_deployed", "hyst", {}))
    for P in (2, 3, 5, 8, 15, 30):
        g.append(("G4_periodic", f"P={P}", dict(P=P)))
    for K in (3, 5, 8, 12):
        g.append(("G5_rate_limited", f"K={K}", dict(K=K)))
    for lam in (0.05, 0.1, 0.2, 0.3):
        g.append(("G6_or_dummy", f"lam={lam}", dict(lam=lam)))
    for K, p1, lam in G7_GRID:
        g.append(("G7_rr", f"K={kname(K)},p1={p1},lam={lam}", dict(K=K, p1=p1, lam=lam)))
    if include_g8:
        for P_r in (8, 15, 30):
            for tau in (0.35, 0.55, 0.72):
                g.append(("G8_ca_refresh", f"Pr={P_r},tau={tau}", dict(P_r=P_r, tau=tau)))
    return g


CORE = {("G1_ca_shield", f"tau={t}") for t in TAUS} | {
    ("G3_deployed", "hyst"), ("G4_periodic", "P=5"), ("G5_rate_limited", "K=5"), ("G6_or_dummy", "lam=0.2"),
    ("G7_rr", "K=5,p1=0.7,lam=0.1"), ("G7_rr", "K=inf,p1=0.5,lam=0.2")}
SWEEP = {("G1_ca_shield", "tau=0.55"), ("G3_deployed", "hyst"), ("G5_rate_limited", "K=5"),
         ("G7_rr", "K=5,p1=0.7,lam=0.1")}


def make(score, valid, family: str, p: dict, rng: np.random.Generator, L: int = 5) -> np.ndarray:
    if family == "G1_ca_shield":
        return content_aware(score, valid, p["tau"], True)
    if family == "G2_ca_pure":
        return content_aware(score, valid, p["tau"], False)
    if family == "G3_deployed":
        return deployed(score, valid)
    if family == "G4_periodic":
        return periodic(len(score), p["P"], int(rng.integers(p["P"])))
    if family == "G5_rate_limited":
        return rate_limited(score, valid, 0.55, p["K"], 25, 5, rng)
    if family == "G6_or_dummy":
        return or_dummy(score, valid, 0.55, 5, 25, p["lam"], rng)
    if family == "G7_rr":
        return rr_certified(score, valid, 0.55, p["K"], 25, p["p1"], p["lam"], rng, L)
    if family == "G8_ca_refresh":
        return ca_refresh(score, valid, p["tau"], p["P_r"], rng)
    raise ValueError(family)
