"""repro_release.py — P4 Q1: what the public C121 release reproduces without the per-frame scores.

Forces C121 to load from the release (data/_derived/39/c121: G3 schedule a_t, event mask, valid flag), checks the
SHA-256 of every array against index.csv, recomputes with the code of attacks.py and the same seeds:
  mi     plug-in (+ permutation null) and KSG MI of G3 (deployed) and G4 periodic (P = 2 ... 30), pooled per tier
  attack A1 / A2-window / A3 attackers on G4 P = 5 (oracle, 1-Hz plug, traffic; W = 1, 2, 5, 10 s), classifier MI bound
  loc    onset localisation (CUSUM, matched filter) on G3 and G4 P = 5
and compares every recomputed row with the committed P1_mi_table.csv, P1_attack_table.csv, P1_attack_loc_table.csv.
G3 presence attackers are trained on the G1 threshold family (threshold-agnostic attacker) and therefore need the
scores, as do G1, G2, G5-G8 on C121, the observer sweep, the iso-energy sweep and the cross-dataset transfer.

Outputs: results/release_check/ (per-part files) and results/release_check/summary.csv. About 1 min on a laptop CPU.
Usage: python repro_release.py
"""
from __future__ import annotations

import hashlib
import os
import time

os.environ["C121_FROM_RELEASE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import attacks as A  # noqa: E402
import gates as G  # noqa: E402
from data_sparse import c121_release_dir, g3, load_corpus  # noqa: E402
from schedules import load_config, results_dir  # noqa: E402

CFG = load_config()
RES = results_dir(CFG)
OUT = RES / "release_check"
CORPUS = "C121"
G4 = [("G4_periodic", f"P={P}", P) for P in (2, 3, 5, 8, 15, 30)]


def check_hashes(clips) -> None:
    idx = pd.read_csv(c121_release_dir(CFG) / "index.csv", dtype={"video": str})
    assert len(idx) == len(clips) == 121
    for c, r in zip(clips, idx.itertuples()):
        assert (c.dataset, c.video) == (r.dataset, r.video)
        for name, x in (("a_g3", g3(c)), ("e", c.e), ("valid", c.valid)):
            h = hashlib.sha256(np.ascontiguousarray(x, dtype=np.int8).tobytes()).hexdigest()
            assert h == getattr(r, f"sha256_{name}"), f"{c.key}: {name} differs from index.csv"
        assert int(g3(c).sum()) == r.n_opens_logged, f"{c.key}: G3 schedule differs from the logged deployed trace"


def schedules(clips) -> dict:
    """Same schedules and seeds as attacks.schedules (G.make), for the gates that need no score."""
    out = {("G3_deployed", "hyst"): [g3(c) for c in clips]}
    for fam, lab, P in G4:
        out[(fam, lab)] = [G.periodic(c.n, P, int(A.seed(CORPUS, fam, lab, c.key).integers(P))) for c in clips]
    return out


def compare(new: pd.DataFrame, ref: pd.DataFrame, keys: list[str], vals: list[str], name: str) -> dict:
    ref = ref[ref.corpus == CORPUS]
    m = new.merge(ref, on=keys, how="left", suffixes=("", "_ref"), indicator=True)
    missing = int((m._merge != "both").sum())
    diff = 0
    worst = 0.0
    for v in vals:
        a, b = m[v].to_numpy(float), m[f"{v}_ref"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        worst = max(worst, float(d.max()) if len(d) else 0.0)
        diff += int((d > 1e-12).sum())
    return dict(table=name, rows=len(new), missing_in_committed=missing, values_differing=diff, max_abs_diff=worst)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    A.OUT = OUT
    clips = load_corpus(CORPUS)
    assert all(c.score is None for c in clips), "C121 must come from the release"
    check_hashes(clips)
    tiers = A.tier_masks(clips)
    sched = schedules(clips)
    print(f"{len(clips)} clips from {c121_release_dir(CFG)}, hashes OK, informative {int(tiers['info'].sum())}",
          flush=True)

    if not (OUT / "mi_C121.parquet").exists():
        A.part_mi(CORPUS, clips, tiers, sched)
    print(f"mi {time.time() - t0:.0f}s", flush=True)
    p5 = {("G4_periodic", "P=5"): sched[("G4_periodic", "P=5")]}
    if not (OUT / "attack_C121.csv").exists():
        A.part_attack(CORPUS, clips, tiers, p5)
    print(f"attack {time.time() - t0:.0f}s", flush=True)
    if not (OUT / "loc_C121.csv").exists():
        A.part_loc(CORPUS, clips, tiers, {("G3_deployed", "hyst"): sched[("G3_deployed", "hyst")], **p5})
    print(f"loc {time.time() - t0:.0f}s", flush=True)

    mi_new = A.mi_summary(pd.read_parquet(OUT / "mi_C121.parquet"))
    lb_new = pd.read_csv(OUT / "milb_C121.csv")
    mi_ref = pd.read_csv(RES / "P1_mi_table.csv")
    k_mi = ["corpus", "family", "param", "observer", "W_s", "target", "tier", "estimator"]
    summary = [
        compare(mi_new, mi_ref, k_mi, ["mean", "ci_lo", "ci_hi", "n_videos", "rho_mean", "H_E_mean"], "P1_mi_table (MI)"),
        compare(lb_new, mi_ref, k_mi, ["mean", "ci_lo", "ci_hi", "n_videos"], "P1_mi_table (classifier bound)"),
        compare(pd.read_csv(OUT / "attack_C121.csv"), pd.read_csv(RES / "P1_attack_table.csv"),
                ["corpus", "family", "param", "observer", "W_s", "target", "model", "tier", "metric"],
                ["value", "ci_lo", "ci_hi", "n_videos", "n_rows"], "P1_attack_table"),
        compare(pd.read_csv(OUT / "loc_C121.csv"), pd.read_csv(RES / "P1_attack_loc_table.csv"),
                ["corpus", "family", "param", "observer", "estimator", "tier"],
                ["MAE_frames", "MAE_ci_lo", "MAE_ci_hi", "P_within_L", "P_ci_lo", "P_ci_hi", "n_onsets"],
                "P1_attack_loc_table"),
    ]
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
