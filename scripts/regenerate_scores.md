# Regenerating the per-frame content scores of C121 (not run here)

The 121-video corpus (CDnet 2014, LASIESTA, BMC) is released in `data/_derived/39/c121/` **without** the per-frame
content score $S_t$: per video, the open schedule of the deployed gate G3 (`a_g3`), the ground-truth in-event
indicator (`e`) and the frame-validity flag (`valid`). The scores themselves (`frame_scores.parquet`, with
`events_gated.csv` and `activation_by_video.csv`) are available from the author on request. This note records how they
were produced, so that they can be regenerated and checked against the release.

## What produced them
`frame_scores.parquet` (columns `dataset, video, frame, score, g_t, valid`) is the output of
`scripts/dump_gate_scores.py` of the certifiable-frame-skipping pipeline behind the author's IEEE IoT-J article
(DOI 10.1109/JIOT.2026.3740931). That script runs the ASMAG+ gate (`ASMAGPlusGate`, asmag-trc sources, commit
`cc3475ea`) once over every frame and writes the raw gate score; the score does not depend on the open threshold.

Inputs:
1. **Frames** of the three public benchmarks, from their providers: CDnet 2014 (changedetection.net), LASIESTA, BMC.
2. **Per-frame detections** (`detections_cpu/<dataset>/<category>__<video>.json`, schema
   `{"frames": [{"frame": int, "boxes": [...]}, ...]}`), from a CPU detector pass over the same frames. They define
   the in-event indicator `g_t` and feed the gate's temporal-persistence input; no detector runs inside the gate pass.
3. **Gate configuration** `asmag_tr_controller_online_guarded_cdnet_full.yaml` (open threshold 0.55, forced refresh
   every 15 frames, reuse cap 4), read by the pipeline, never hard-coded.

## Steps
1. Point the pipeline's `configs/default.yaml` (`paths.detections_root`, `paths.frames_root_*`, `paths.asmag_src`,
   `paths.asmag_config`) to the inputs above.
2. `python scripts/dump_gate_scores.py` (no detector inside the gate pass) writes
   `data/processed/frame_scores.parquet`; `python scripts/generate_gate_scores.py` writes `events_gated.csv`.
3. In this repository, set `CFS_PROCESSED_DIR` to that `data/processed` folder.

## Check against the release
A necessary check on regenerated scores: the deployed-gate replay must reproduce the released schedules bit for bit
(the export also needs `activation_by_video.csv` from `generate_gate_scores.py`).

```bash
cd code
CFS_PROCESSED_DIR=<.../data/processed> python export_c121_release.py --out ../c121_regenerated
```

then compare the `sha256_a_g3`, `sha256_e` and `sha256_valid` columns of `../c121_regenerated/index.csv` with
`data/_derived/39/c121/index.csv` (the export also asserts that each G3 schedule has the open count of the logged deployed trace). With matching hashes,
`attacks.py`, `rho_sweep.py` and `verify_bounds.py` recompute the score-dependent tables (threshold sweeps G1/G2,
rate-limited and randomized-response gates G5–G7, observer sweep, iso-energy sweep, cross-dataset transfer).
