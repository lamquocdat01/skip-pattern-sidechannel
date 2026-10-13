"""data_sparse.py — T08: the three corpora of P1 behind one loader (PREREG_39 §2).

  C121   CDnet2014 + LASIESTA + BMC from frame_scores.parquet (stress case, pi ~ 0.77), fps 25 (assumed)
  VIRAT  8 annotated VIRAT clips; score from virat_scores.py dumps; E = E^disc (discrete activities) or E^veh
  SYN    synthetic sparse-event MODEL (not data): pi in {0.02, 0.05, 0.1, 0.2, 0.5}, 30 videos x 15 000 frames,
         D ~ empirical events_gated.csv durations, geometric gaps, score = Gaussian-copula AR(1) mapped through the
         empirical in/out-of-event score distributions of C121.

`python data_sparse.py` fits the synthetic model (results/P1_synthetic_model.json) and prints a corpus summary;
`python data_sparse.py v2` fits the post-hoc v2 model (P1b T14, off-event marginal from VIRAT) to
results/P1b_synthetic_model_v2.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from schedules import load_config, load_videos, processed_dir, results_dir
from virat_scores import virat_paths


@dataclass
class Clip:
    corpus: str              # C121 | VIRAT | SYN_pi0.05 ...
    dataset: str             # CDnet2014 / LASIESTA / BMC / VIRAT / SYN
    video: str
    fps: float
    score: np.ndarray
    valid: np.ndarray
    e: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.score)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.video}"


# ------------------------------------------------------------------ C121
def load_c121(cfg) -> list[Clip]:
    return [Clip("C121", v.dataset, v.video, float(cfg["fps"][v.dataset]), v.score.astype(np.float64),
                 v.valid, v.e) for v in load_videos(cfg)]


# ------------------------------------------------------------------ VIRAT
def load_virat(cfg, target: str = "disc") -> list[Clip]:
    _, _, ddir = virat_paths(cfg)
    out = []
    for f in sorted(ddir.glob("*.npz")):
        z = np.load(f)
        e = z["e_disc"] if target == "disc" else z["e_veh"]
        out.append(Clip("VIRAT" if target == "disc" else "VIRAT_veh", "VIRAT", f.stem, float(cfg["virat"]["fps"]),
                        z["score"].astype(np.float64), z["valid"].astype(np.int8), e.astype(np.int8)))
    return out


# ------------------------------------------------------------------ SYN
def fit_synthetic_model(cfg, rng: np.random.Generator) -> dict:
    c121 = load_c121(cfg)
    s = np.concatenate([c.score[c.valid == 1] for c in c121])
    e = np.concatenate([c.e[c.valid == 1] for c in c121])
    q = np.linspace(0, 1, 1001)
    q0, q1 = np.quantile(s[e == 0], q), np.quantile(s[e == 1], q)
    # lag-1 autocorrelation of normal scores within runs of constant E (random tie-break for point masses)
    zs, zprev = [], []
    for c in c121:
        m = c.valid == 1
        sc, ee = c.score[m], c.e[m]
        u = np.empty(len(sc))
        for lab, qq in ((0, q0), (1, q1)):
            k = ee == lab
            lo = np.searchsorted(qq, sc[k], side="left") / 1000.0
            hi = np.searchsorted(qq, sc[k], side="right") / 1000.0
            u[k] = lo + (hi - lo) * rng.random(k.sum())
        z = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
        same = ee[1:] == ee[:-1]
        zs.append(z[1:][same]); zprev.append(z[:-1][same])
    phi = float(np.corrcoef(np.concatenate(zs), np.concatenate(zprev))[0, 1])
    ev = pd.read_csv(processed_dir(cfg) / cfg["data"]["events_gated"])
    ev = ev[ev.dataset.isin(cfg["data"]["datasets"])]
    D = ev["duration"].to_numpy(int)
    D = D[D >= 1]
    return dict(phi=phi, q_grid=q.tolist(), q0=q0.tolist(), q1=q1.tolist(), durations=D.tolist(),
                mean_D=float(D.mean()), median_D=float(np.median(D)), frac_D_lt_5=float((D < 5).mean()),
                n_events=int(len(D)), score_mean_out=float(s[e == 0].mean()), score_mean_in=float(s[e == 1].mean()),
                note="MODEL, not data: Gaussian-copula AR(1) score through empirical C121 marginals; "
                     "D resampled from events_gated.csv; geometric gaps with mean mean_D*(1-pi)/pi")


def fit_synthetic_model_v2(cfg, base: dict) -> dict:
    """P1b T14 (POST-HOC after CP-B1): same model as v1 except the off-event score marginal q0, which is fitted on
    the off-event (E^disc = 0) valid frames of the 8 VIRAT clips instead of C121. phi, q1, D unchanged."""
    vir = load_virat(cfg, "disc")
    s0 = np.concatenate([c.score[(c.valid == 1) & (c.e == 0)] for c in vir])
    m = dict(base)
    m["q0"] = np.quantile(s0, np.asarray(base["q_grid"])).tolist()
    m["score_mean_out"] = float(s0.mean())
    m["n_off_frames_virat"] = int(len(s0))
    m["note"] = ("MODEL v2 (post-hoc, P1b T14): v1 with the off-event score marginal q0 refitted on VIRAT off-event "
                 "frames; phi, q1 and D from C121 as in v1")
    return m


def synth_events(T: int, pi: float, D: np.ndarray, rng) -> np.ndarray:
    e = np.zeros(T, np.int8)
    gap_mean = max(1.0, D.mean() * (1 - pi) / pi)
    t = int(rng.geometric(1.0 / gap_mean))
    while t < T:
        d = int(rng.choice(D))
        e[t:t + d] = 1
        t += d + int(rng.geometric(1.0 / gap_mean))
    return e


def synth_score(e: np.ndarray, model: dict, rng) -> np.ndarray:
    T, phi = len(e), model["phi"]
    eps = rng.standard_normal(T)
    z = np.empty(T)
    z[0] = eps[0]
    s = np.sqrt(1 - phi ** 2)
    for t in range(1, T):            # T = 15 000, cheap enough
        z[t] = phi * z[t - 1] + s * eps[t]
    u = norm.cdf(z)
    g = np.asarray(model["q_grid"])
    return np.where(e == 1, np.interp(u, g, model["q1"]), np.interp(u, g, model["q0"]))


def load_syn(cfg, pis=None, model: dict | None = None) -> list[Clip]:
    sc = cfg["synthetic"]
    pis = pis or sc["pi_sweep"]
    if model is None:
        model = json.loads((results_dir(cfg) / "P1_synthetic_model.json").read_text(encoding="utf-8"))
    D = np.asarray(model["durations"])
    out = []
    for pi in pis:
        rng = np.random.default_rng([42, int(round(pi * 1000))])
        for i in range(sc["n_videos"]):
            e = synth_events(sc["T"], pi, D, rng)
            out.append(Clip(f"SYN_pi{pi}", "SYN", f"pi{pi}_v{i:02d}", float(sc["fps"]), synth_score(e, model, rng),
                            np.ones(sc["T"], np.int8), e, dict(pi=pi)))
    return out


def load_corpus(name: str, cfg=None) -> list[Clip]:
    cfg = cfg or load_config()
    if name == "C121":
        return load_c121(cfg)
    if name == "VIRAT":
        return load_virat(cfg, "disc")
    if name == "VIRAT_veh":
        return load_virat(cfg, "veh")
    if name.startswith("SYN_pi"):
        return load_syn(cfg, [float(name[6:])])
    raise ValueError(name)


def main() -> None:
    import sys
    cfg = load_config()
    if sys.argv[1:] == ["v2"]:
        base = json.loads((results_dir(cfg) / "P1_synthetic_model.json").read_text(encoding="utf-8"))
        model = fit_synthetic_model_v2(cfg, base)
        (results_dir(cfg) / "P1b_synthetic_model_v2.json").write_text(json.dumps(model), encoding="utf-8")
    else:
        model = fit_synthetic_model(cfg, np.random.default_rng(42))
        (results_dir(cfg) / "P1_synthetic_model.json").write_text(json.dumps(model), encoding="utf-8")
    print(f"phi = {model['phi']:.3f}, mean D = {model['mean_D']:.1f}, median D = {model['median_D']}, "
          f"frac D<5 = {model['frac_D_lt_5']:.2f}, n = {model['n_events']}")
    rows = []
    for pi in cfg["synthetic"]["pi_sweep"]:
        clips = load_syn(cfg, [pi], model)
        rows.append(dict(corpus=f"SYN_pi{pi}", n_videos=len(clips), pi_realised=np.mean([c.e.mean() for c in clips]),
                         events_per_video=np.mean([(np.diff(np.r_[0, c.e]) == 1).sum() for c in clips]),
                         score_in=np.mean([c.score[c.e == 1].mean() for c in clips if c.e.any()]),
                         score_out=np.mean([c.score[c.e == 0].mean() for c in clips])))
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
