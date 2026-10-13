"""make_replay_schedules.py — T22a B1: write the on-device replay schedules (LittleFS image source).

For each selected clip the open schedule a_t of the deployed gate G3 (exact ASMAG+ replay, gates.deployed) is written
as fw_replay/data/sched_<tag>.csv (columns frame,a_t), plus fw_replay/data/config.json for the firmware and
results/real_trace/replay_selection.csv (provenance: n, fps, rho, pi, evaluable windows, sha256 of each schedule).

Clip choice (T22: 1 CDnet evaluable, 1 LASIESTA, 1 VIRAT) — all three in the "informative" tier (G3 rate < 0.6):
  CDnet2014/blizzard     A1-evaluable at W = 1, 2, 5 s, pi 0.37, 6100 frames
  LASIESTA/O_RA_02       no LASIESTA clip is A1-evaluable (all < 30 windows); O_RA_02 is the informative clip with
                         both event and non-event frames (pi 0.34), 375 frames
  VIRAT/VIRAT_S_000002   A1-evaluable at W = 1, 2, 5, 10 s, pi 0.22, 9075 frames

Usage: python make_replay_schedules.py            (from code/real_trace/)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import attacks as A  # noqa: E402
from data_sparse import load_c121, load_virat  # noqa: E402
from schedules import load_config, results_dir  # noqa: E402

SELECTION = [  # (tag, dataset, video)
    ("cdnet_blizzard", "CDnet2014", "blizzard"),
    ("lasiesta_O_RA_02", "LASIESTA", "O_RA_02"),
    ("virat_S_000002", "VIRAT", "VIRAT_S_000002"),
]

# Replay timing (T22a prompt B1). busy_ms ~ one C2b inference at 13.9 fps; fps_play must leave the busy loop inside
# one frame period, so all clips replay at 12.5 frames/s (80 ms period) and the time axis is dilated by
# fps_native / fps_play (2.0 for 25 fps, 2.4 for VIRAT 30 fps). Every sample carries the frame index, so T22b maps
# wall time back to video time exactly.
REPLAY = dict(fps_play=12.5, busy_ms=72, sample_hz=10, warmup_s=10, idle_s=20, min_shunt_ma=15.0, max_total_min=45.0)


def main() -> None:
    cfg = load_config()
    clips = load_c121(cfg) + load_virat(cfg)
    tiers = A.tier_masks(clips)
    data_dir = HERE / "fw_replay" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in data_dir.glob("sched_*.csv"):
        f.unlink()

    rows, sched_cfg = [], []
    for tag, ds, video in SELECTION:
        i = next(k for k, c in enumerate(clips) if c.dataset == ds and c.video == video)
        c = clips[i]
        a = A.G.deployed(c.score, c.valid).astype(np.int8)
        text = "frame,a_t\n" + "".join(f"{k},{int(v)}\n" for k, v in enumerate(a))
        name = f"sched_{tag}.csv"
        (data_dir / name).write_bytes(text.encode("ascii"))
        ev = [W for W in A.W_S if tiers["ev"][(i, W, "A1")]]
        rows.append(dict(tag=tag, dataset=ds, video=video, n_frames=c.n, fps_native=c.fps,
                         dur_native_s=round(c.n / c.fps, 3), rho_G3=round(float(a.mean()), 6), n_open=int(a.sum()),
                         pi=round(float(c.e.mean()), 6), informative=bool(tiers["info"][i]),
                         A1_evaluable_W_s=";".join(map(str, ev)), dilation=round(c.fps / REPLAY["fps_play"], 4),
                         replay_s=round(c.n / REPLAY["fps_play"], 1),
                         sha256=hashlib.sha256(text.encode("ascii")).hexdigest()))
        sched_cfg.append(dict(tag=tag, file=f"/{name}", n=int(c.n), n_open=int(a.sum()), fps_native=c.fps))

    total_s = REPLAY["warmup_s"] + sum(r["replay_s"] + REPLAY["idle_s"] for r in rows)
    assert total_s / 60 <= REPLAY["max_total_min"], f"replay too long: {total_s / 60:.1f} min"
    conf = dict(version="T22a-1", gate="G3_deployed", **{k: v for k, v in REPLAY.items() if k != "max_total_min"},
                expected_total_s=round(total_s, 1), schedules=sched_cfg)
    (data_dir / "config.json").write_text(json.dumps(conf, indent=1), encoding="ascii")

    out = results_dir(cfg) / "real_trace"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "replay_selection.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="sha256").to_string(index=False))
    print(f"expected run time {total_s:.0f} s = {total_s / 60:.1f} min")


if __name__ == "__main__":
    main()
