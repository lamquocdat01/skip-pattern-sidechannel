"""mi_estimators.py — the three MI estimators of PREREG_39 §6 (all in bits).

1. plug-in binned I(E; N_W), N_W in <= 8 equal-width bins of [0, 1] (rate per frame), bias-corrected by the mean of a
   within-video permutation null (n_shuffle shuffles of E across windows); p = share of null >= observed.
2. KSG-type kNN estimator for a discrete target vs a continuous feature (Ross 2014) via
   sklearn.feature_selection.mutual_info_classif (k = 3), on the per-window open rate + 1e-6 jitter.
3. classifier lower bound I >= H(E_W) - CE_test, CE = held-out cross-entropy (bits) of any probabilistic classifier.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

N_BINS = 8
LOG2E = 1.0 / np.log(2.0)


def h2(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.where(p > 0, p, 1.0)
    return -(p * np.log2(p)).sum(-1)


def mi_batch(x: np.ndarray, y: np.ndarray, ny: int) -> np.ndarray:
    """Plug-in I(X;Y) for binary rows x (S, n) against one symbol vector y (n,)."""
    S, n = x.shape
    idx = x.astype(np.int64) * ny + y[None, :] + (np.arange(S)[:, None] * 2 * ny)
    joint = np.bincount(idx.ravel(), minlength=S * 2 * ny).reshape(S, 2, ny) / n
    return _entropy_bits(joint.sum(2)) + _entropy_bits(joint.sum(1)) - _entropy_bits(joint.reshape(S, -1))


def bin_rate(rate: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(rate * N_BINS).astype(int), 0, N_BINS - 1)


def plugin(e: np.ndarray, rate: np.ndarray, rng, n_shuffle: int = 200):
    """Returns (MI_plugin, MI_null_mean, MI_corrected, p_value)."""
    n = len(e)
    if n < 4 or e.min() == e.max():
        return 0.0, 0.0, 0.0, 1.0
    y = bin_rate(rate)
    obs = float(mi_batch(e[None, :], y, N_BINS)[0])
    perm = np.argsort(rng.random((n_shuffle, n)), axis=1)
    null = mi_batch(e[perm], y, N_BINS)
    nm = float(null.mean())
    return obs, nm, max(0.0, obs - nm), float((null >= obs - 1e-12).mean())


def ksg(e: np.ndarray, rate: np.ndarray, rng, k: int = 3) -> float:
    if len(e) < 2 * (k + 1) or e.min() == e.max() or np.bincount(e).min() <= k:
        return 0.0
    x = (rate + 1e-6 * rng.standard_normal(len(rate)))[:, None]
    return float(mutual_info_classif(x, e, discrete_features=False, n_neighbors=k,
                                     random_state=int(rng.integers(2**31)))[0] * LOG2E)


def classifier_lb(e: np.ndarray, prob: np.ndarray) -> float:
    """H(E) - CE(prob) in bits, clipped at 0 (valid lower bound on I(E; features))."""
    if len(e) == 0 or e.min() == e.max():
        return 0.0
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    ce = -np.mean(e * np.log2(p) + (1 - e) * np.log2(1 - p))
    return max(0.0, h2(float(e.mean())) - float(ce))
