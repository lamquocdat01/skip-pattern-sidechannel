"""observer.py — what a passive observer sees of an open schedule a_t.

Two levels (threat model §1, theory_notes.md):
  oracle : the observer sees a_t exactly (upper bound on any physical channel; data-processing ⇒
           I(E; O) <= I(E; A) for every O that is a function of A plus independent noise).
  noisy  : power channel  — wall plug / shunt sampling mean power at fs Hz, Gaussian noise sigma,
                            quantisation step q;
           traffic channel — one encrypted burst per open (size ~ N(mu, sd)), timestamp jitter;
                            the observer keeps only timing and size.

Energy constants (mJ/frame) come from config.yaml (Deep3 C2b values, simulator parameters only).
"""
from __future__ import annotations

import numpy as np


def oracle(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int8)


def power_trace(a: np.ndarray, fps: float, ocfg: dict, fs_hz: float, sigma_w: float,
                rng: np.random.Generator) -> np.ndarray:
    """Mean power (W) per sampling interval of 1/fs seconds.

    Per-frame energy is E_idle or E_active (mJ); mean power over an interval is the energy in the
    interval divided by its duration, plus the device baseline P_base. Trailing partial interval
    is dropped."""
    e_frame = np.where(a > 0, ocfg["E_active_mJ"], ocfg["E_idle_mJ"]) * 1e-3   # J per frame
    fpi = max(1, int(round(fps / fs_hz)))                                   # frames per interval
    m = len(e_frame) // fpi
    p = e_frame[: m * fpi].reshape(m, fpi).sum(1) / (fpi / fps) + ocfg["P_base_W"]
    p = p + rng.normal(0.0, sigma_w, m) if sigma_w > 0 else p
    q = ocfg["power"]["quant_W"]
    return np.round(p / q) * q if q else p


def power_to_open_count(p: np.ndarray, fps: float, ocfg: dict, fs_hz: float) -> np.ndarray:
    """Invert the power model: estimated number of opens per sampling interval (real-valued)."""
    fpi = max(1, int(round(fps / fs_hz)))
    dt = fpi / fps
    e_int = (p - ocfg["P_base_W"]) * dt                      # J in interval
    e_idle, e_act = ocfg["E_idle_mJ"] * 1e-3, ocfg["E_active_mJ"] * 1e-3
    return (e_int - fpi * e_idle) / (e_act - e_idle)


def traffic_trace(a: np.ndarray, fps: float, ocfg: dict, rng: np.random.Generator):
    """Encrypted-traffic view: (timestamps_s, sizes_bytes) of one burst per open."""
    t_open = np.flatnonzero(a) / fps
    tc = ocfg["traffic"]
    ts = t_open + rng.normal(0.0, tc["jitter_ms_sd"] * 1e-3, len(t_open))
    sz = np.maximum(64, rng.normal(tc["pkt_bytes_mean"], tc["pkt_bytes_sd"], len(t_open))).round()
    order = np.argsort(ts)
    return ts[order], sz[order]


def traffic_to_open_count(ts: np.ndarray, n_frames: int, fps: float, win_frames: int) -> np.ndarray:
    """Bursts per window of win_frames — what a timing-only observer recovers."""
    edges = np.arange(0, n_frames // win_frames + 1) * win_frames / fps
    return np.histogram(ts, bins=edges)[0]
