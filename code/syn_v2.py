"""syn_v2.py — P1b T14 (POST-HOC after CP-B1): MI oracle on SYN with the v2 score model.

v2 = v1 with the off-event score marginal refitted on VIRAT off-event frames (data_sparse.fit_synthetic_model_v2).
Only A1, oracle, tier all, W in {1, 10} s, pi in {0.02, 0.05, 0.1}; gates G1 tau = 0.55 (criterion-3 cell), G3, G4 P = 5.
v1 is recomputed in the same run for a like-for-like comparison. PREREG criterion 3 (locked, v1) stays NOT MET.

Output: results/P1b_syn_v2_mi.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import gates as G
import mi_estimators as M
from data_sparse import load_syn
from schedules import load_config, results_dir

CFG = load_config()
L = CFG["certificate_L"]
OUT = results_dir(CFG)
PIS = [0.02, 0.05, 0.1]
W_S = [1, 10]
GATES = [("G1_ca_shield", "tau=0.55", dict(tau=0.55)), ("G3_deployed", "hyst", {}), ("G4_periodic", "P=5", dict(P=5))]
N_SHUFFLE = 200
N_BOOT = 2000


def main():
    models = {"v1": json.loads((OUT / "P1_synthetic_model.json").read_text(encoding="utf-8")),
              "v2": json.loads((OUT / "P1b_synthetic_model_v2.json").read_text(encoding="utf-8"))}
    rows = []
    for mname, model in models.items():
        for pi in PIS:
            clips = load_syn(CFG, [pi], model)
            for fam, lab, p in GATES:
                for W_s in W_S:
                    mi, he, rho = [], [], []
                    for i, c in enumerate(clips):
                        rng = np.random.default_rng([77, int(pi * 1000), i, W_s, GATES.index((fam, lab, p))])
                        a = G.make(c.score, c.valid, fam, p, rng, L)
                        w = int(round(W_s * c.fps))
                        m = c.n // w
                        e = c.e[: m * w].reshape(m, w).max(1).astype(np.int8)
                        rate = a[: m * w].reshape(m, w).mean(1)
                        mi.append(M.plugin(e, rate, rng, N_SHUFFLE)[2])
                        he.append(M.h2(float(e.mean())))
                        rho.append(float(a.mean()))
                    mi = np.asarray(mi)
                    b = mi[np.random.default_rng([78, int(pi * 1000), W_s]).integers(len(mi), size=(N_BOOT, len(mi)))].mean(1)
                    r = dict(model=mname, pi=pi, family=fam, param=lab, W_s=W_s, n_videos=len(clips),
                             rho=float(np.mean(rho)), H_E=float(np.mean(he)), MI_corr=float(mi.mean()),
                             MI_lo=float(np.quantile(b, 0.025)), MI_hi=float(np.quantile(b, 0.975)),
                             MI_over_H=float(mi.mean() / np.mean(he)) if np.mean(he) > 0 else np.nan,
                             score_out=float(np.mean([c.score[c.e == 0].mean() for c in clips])))
                    rows.append(r)
                    print(f"{mname} pi={pi} {fam:12s} W={W_s:2d}s rho={r['rho']:.3f} H={r['H_E']:.3f} "
                          f"MI={r['MI_corr']:.4f} [{r['MI_lo']:.4f},{r['MI_hi']:.4f}]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "P1b_syn_v2_mi.csv", index=False)


if __name__ == "__main__":
    main()
