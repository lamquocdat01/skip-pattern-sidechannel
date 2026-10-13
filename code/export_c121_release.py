"""export_c121_release.py — P4 Q1: public release of the C121 corpus without per-frame scores.

Written for every one of the 121 videos (needs the per-frame scores, CFS_PROCESSED_DIR):
  <datasets root>/_derived/39/c121/<dataset>/<video>.npz
      a_g3   int8  open schedule of the deployed gate G3 (exact ASMAG+ replay, gates.deployed)
      e      int8  ground-truth in-event indicator per frame (g_t of frame_scores.parquet)
      valid  int8  frame validity flag
  <datasets root>/_derived/39/c121/index.csv
      dataset, video, file, frame0, n_frames, fps, n_opens_g3, n_opens_logged (deployed trace of activation_by_video.csv),
      rho_g3, pi, sha256 of each array (raw int8 bytes)
The scores S_t, events_gated.csv and activation_by_video.csv themselves are not released (README, "Data").
With this release, data_sparse.load_c121 returns clips with score = None: G3 and every score-free gate (periodic G4)
can be evaluated; gates that threshold S_t cannot.

Usage: python export_c121_release.py [--out DIR]   (default: the release folder above)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import gates as G
from data_sparse import c121_release_dir
from schedules import load_config, load_videos, processed_dir


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.int8).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    cfg = load_config()
    out = Path(ap.parse_args().out or c121_release_dir(cfg))
    act = pd.read_csv(processed_dir(cfg) / cfg["data"]["activation_by_video"]).set_index(["dataset", "video"])
    rows = []
    for v in load_videos(cfg):
        a = G.deployed(v.score, v.valid).astype(np.int8)
        e, valid = v.e.astype(np.int8), v.valid.astype(np.int8)
        logged = int(act.loc[(v.dataset, v.video), "n_opens_deployed"])
        assert int(a.sum()) == logged, f"{v.key}: G3 replay {int(a.sum())} != logged {logged}"
        rel = f"{v.dataset}/{v.video}.npz"
        (out / v.dataset).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / rel, a_g3=a, e=e, valid=valid)
        rows.append(dict(dataset=v.dataset, video=v.video, file=rel, frame0=v.frame0, n_frames=v.n,
                         fps=float(cfg["fps"][v.dataset]), n_opens_g3=int(a.sum()), n_opens_logged=logged,
                         rho_g3=round(float(a.mean()), 6), pi=round(float(e.mean()), 6),
                         sha256_a_g3=sha(a), sha256_e=sha(e), sha256_valid=sha(valid)))
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False, lineterminator="\n")
    n = int(idx.n_frames.sum())
    print(f"{len(idx)} videos, {n} frames -> {out}; G3 rate {idx.n_opens_g3.sum() / n:.4f}, "
          f"pi {float((idx.pi * idx.n_frames).sum() / n):.4f}")


if __name__ == "__main__":
    main()
