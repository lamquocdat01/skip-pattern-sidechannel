"""schedules.py — load the 121-video gate trace and generate open schedules a_t for 4 gate families.

Inputs (paths from config.yaml / env vars, never hard-coded):
  frame_scores.parquet : per-frame gate score S_t, in-event indicator g_t (ground truth E_t), valid flag
  events_gated.csv     : per-event onset / duration (used for onset targets and a consistency check)

Gate families (a_t in {0,1}):
  content_aware(tau, shield)   open iff S_t > tau  [OR forced refresh OR reuse cap, if shield]
                               shield=True reproduces the deployed ASMAG+ logic (certified latency L = W_maxskip+1)
  periodic(P, phase)           open iff (t - phase) mod P == 0  — oblivious, independent of content
  rate_limited(tau, K, Wb)     content demand S_t > tau, admitted only if < K content opens in the last Wb
                               frames; plus an oblivious periodic floor (period floor_P) that is not rate-limited
  privacy_certified(..., lam)  rate_limited OR Bernoulli(lam) dummy activation on every remaining frame
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parent
TOPIC_DIR = CODE_DIR.parent


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CODE_DIR / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def processed_dir(cfg: dict) -> Path:
    d = cfg["data"]
    if os.environ.get(d["processed_dir_env"]):
        return Path(os.environ[d["processed_dir_env"]])
    root = os.environ.get(d["repo_root_env"]) or d["repo_root"]
    if not root:
        raise RuntimeError(f"C121 per-frame scores not configured: set {d['processed_dir_env']} to the folder holding "
                           f"{d['frame_scores']} (see README, 'Data')")
    return Path(root) / d["processed_subdir"]


def results_dir(cfg: dict) -> Path:
    p = TOPIC_DIR / cfg["output"]["results_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Video:
    dataset: str
    video: str
    frame0: int
    score: np.ndarray            # float32, S_t
    valid: np.ndarray            # int8
    e: np.ndarray                # int8, in-event indicator g_t (ground-truth E_t)
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # positions (0-based)

    @property
    def n(self) -> int:
        return len(self.score)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.video}"


def load_videos(cfg: dict) -> list[Video]:
    pdir = processed_dir(cfg)
    fs = pd.read_parquet(pdir / cfg["data"]["frame_scores"])
    ev = pd.read_csv(pdir / cfg["data"]["events_gated"])
    keep = set(cfg["data"]["datasets"])
    fs = fs[fs["dataset"].isin(keep)]
    ev = ev[ev["dataset"].isin(keep)]
    ev_groups = {k: g for k, g in ev.groupby(["dataset", "video"])}
    out = []
    for (ds, vid), g in fs.groupby(["dataset", "video"], sort=True):
        g = g.sort_values("frame")
        f0 = int(g["frame"].iloc[0])
        eg = ev_groups.get((ds, str(vid)))
        onsets = np.zeros(0, int) if eg is None else np.sort(eg["onset"].to_numpy(int) - f0)
        out.append(Video(ds, str(vid), f0, g["score"].to_numpy(np.float32),
                         g["valid"].to_numpy(np.int8), g["g_t"].to_numpy(np.int8), onsets))
    return out


def event_runs(e: np.ndarray) -> np.ndarray:
    """Maximal runs of e == 1 as (start, end_inclusive) rows."""
    d = np.diff(np.concatenate([[0], e.astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1], axis=1)


# ---------------------------------------------------------------- gate families
def content_aware(v: Video, tau: float, shield: bool = True, T_refresh: int = 15,
                  W_maxskip: int = 4, reuse_logic: str = "deployed") -> np.ndarray:
    """Replays certifiable-frame-skipping/scripts/sweep_tau.py::replay_opens when shield=True."""
    demand = (v.score > tau) & (v.valid == 1)
    if not shield:
        return demand.astype(np.int8)
    a = np.zeros(v.n, np.int8)
    fsl = W_maxskip + 1  # forces an open on the first frame, as in the generator
    fixed = reuse_logic == "fixed"
    for t in range(v.n):
        if not v.valid[t]:
            fsl += 1
            continue
        cap = fsl >= W_maxskip if fixed else fsl > W_maxskip
        o = demand[t] or (t > 0 and t % T_refresh == 0) or cap
        a[t] = o
        fsl = 0 if o else fsl + 1
    return a


def periodic(n: int, P: int, phase: int = 0) -> np.ndarray:
    a = np.zeros(n, np.int8)
    a[phase % P::P] = 1
    return a


def rate_limited(v: Video, tau: float, K: int, Wb: int, floor_P: int | None = 5) -> np.ndarray:
    """Token-bucket style: a content open is admitted only if fewer than K content opens fell in the
    last Wb frames (sliding window). The periodic floor keeps a latency certificate and is oblivious,
    so it is not charged to the bucket."""
    demand = (v.score > tau) & (v.valid == 1)
    floor = periodic(v.n, floor_P) if floor_P else np.zeros(v.n, np.int8)
    c = np.zeros(v.n, np.int8)       # admitted content opens
    run = 0                          # content opens in window (t-Wb, t-1]
    for t in range(v.n):
        if t >= Wb:
            run -= c[t - Wb]
        if demand[t] and not floor[t] and run < K:
            c[t] = 1
            run += 1
    return np.maximum(c, floor)


def privacy_certified(v: Video, tau: float, K: int, Wb: int, lam: float,
                      rng: np.random.Generator, floor_P: int | None = 5) -> np.ndarray:
    a = rate_limited(v, tau, K, Wb, floor_P)
    dummy = (rng.random(v.n) < lam).astype(np.int8)
    return np.maximum(a, dummy)


def family_grid(cfg: dict):
    """Yield (family, param_label, params) for every schedule configured in config.yaml."""
    gf, dg = cfg["gate_families"], cfg["deployed_gate"]
    for shield in gf["content_aware"]["shield"]:
        for tau in gf["content_aware"]["tau_sweep"]:
            fam = "content_aware_shield" if shield else "content_aware_pure"
            yield fam, f"tau={tau}", dict(tau=tau, shield=shield, T_refresh=dg["T_refresh"],
                                          W_maxskip=dg["W_maxskip"], reuse_logic=dg["reuse_logic"])
    for P in gf["periodic"]["P_sweep"]:
        yield "periodic", f"P={P}", dict(P=P)
    rl = gf["rate_limited"]
    for K in rl["K_sweep"]:
        yield "rate_limited", f"K={K}/W={rl['Wb']}", dict(tau=rl["tau"], K=K, Wb=rl["Wb"])
    pc = gf["privacy_certified"]
    for lam in pc["lambda_d_sweep"]:
        yield "privacy_certified", f"lam={lam}", dict(tau=pc["tau"], K=pc["K"], Wb=pc["Wb"], lam=lam)


def make_schedule(v: Video, family: str, p: dict, rng: np.random.Generator) -> np.ndarray:
    if family.startswith("content_aware"):
        return content_aware(v, **p)
    if family == "periodic":
        return periodic(v.n, p["P"], phase=int(rng.integers(p["P"])))  # random phase, independent of E
    if family == "rate_limited":
        return rate_limited(v, **p)
    if family == "privacy_certified":
        return privacy_certified(v, rng=rng, **p)
    raise ValueError(family)


if __name__ == "__main__":
    cfg = load_config()
    vids = load_videos(cfg)
    print(f"{len(vids)} videos, {sum(v.n for v in vids)} frames from {processed_dir(cfg)}")
    # consistency: g_t runs vs events_gated rows
    n_runs = sum(len(event_runs(v.e)) for v in vids)
    n_ev = sum(len(v.onsets) for v in vids)
    print(f"g_t runs = {n_runs}, events_gated rows = {n_ev}")
    dg = cfg["deployed_gate"]
    act = pd.read_csv(processed_dir(cfg) / cfg["data"]["activation_by_video"]).set_index(["dataset", "video"])
    # The threshold replay reproduces scripts/sweep_tau.py (a = 0.711 at theta = 0.5446), NOT the
    # deployed trace (a = 0.770): the deployed ASMAG gate keeps hysteresis state (gate_open) that a
    # pure S_t > tau rule does not capture. We study the replayable family and say so.
    opens = sum(int(content_aware(v, dg["tau"], True, dg["T_refresh"], dg["W_maxskip"],
                                  dg["reuse_logic"]).sum()) for v in vids)
    n = sum(v.n for v in vids)
    ref = act.loc[[(v.dataset, v.video) for v in vids], "n_opens_deployed"].sum()
    print(f"open rate: threshold replay tau={dg['tau']} -> {opens / n:.4f}; deployed trace -> {ref / n:.4f}")
