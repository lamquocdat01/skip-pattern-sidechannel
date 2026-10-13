"""virat_scores.py — T08(a): gate scores + event ground truth for the VIRAT clips that have DIVA annotations.

For every clip in THS_DATASETS/VIRAT/videos with a matching *.activities.yml in VIRAT/annotations_diva:
  * runs ASMAGPlusGate (read-only import from the ASMAG source tree, as certifiable-frame-skipping/
    scripts/generate_gate_scores.py does) on frames resized to 320 px;
  * temporal-persistence feedback uses "a detection exists" = >= 1 annotated geom box in the frame
    (a proxy for the YOLO detections used on the other corpora — declared in PREREG §2);
  * builds E^disc (union of discrete activities, PREREG §2) and E^veh (union of vehicle_* activities).
One worker; stops if free RAM < min_free_ram_gb. Dump (npz per clip) -> THS_DATASETS/_derived/39/virat/.
Small inventory -> results/P1_virat_inventory.csv.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
import yaml

from schedules import load_config, results_dir

ACT_RE = re.compile(r"act2: \{ (\w+): [\d.]+ \}.*?timespan: \[\{tsr0: \[(\d+), (\d+)\]\}\]")
GEOM_RE = re.compile(r"geom: \{ id1: \d+, id0: \d+, ts0: (\d+),")


def datasets_root(cfg: dict) -> Path:
    p = Path(os.environ.get(cfg["datasets_root_env"]) or cfg["datasets_root_default"])
    return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p   # relative = to the topic/repo root


def virat_paths(cfg: dict):
    root = datasets_root(cfg)
    vc = cfg["virat"]
    return root / vc["videos_subdir"], root / vc["annotations_subdir"], root / vc["derived_subdir"]


def find_annotation(ann_dir: Path, clip: str, kind: str) -> Path | None:
    hits = list(ann_dir.glob(f"*/{clip}.{kind}.yml"))
    return hits[0] if hits else None


def activity_masks(act_file: Path, n: int, disc: set[str]):
    e_disc = np.zeros(n, np.int8)
    e_veh = np.zeros(n, np.int8)
    names = []
    with open(act_file, encoding="utf-8") as fh:
        for line in fh:
            m = ACT_RE.search(line)
            if not m:
                continue
            a, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            names.append(a)
            if a in disc:
                e_disc[s:min(e + 1, n)] = 1
            if a.startswith("vehicle_"):
                e_veh[s:min(e + 1, n)] = 1
    return e_disc, e_veh, names


def detection_mask(geom_file: Path, n: int) -> np.ndarray:
    has = np.zeros(n, bool)
    with open(geom_file, encoding="utf-8") as fh:
        for line in fh:
            m = GEOM_RE.search(line)
            if m:
                t = int(m.group(1))
                if t < n:
                    has[t] = True
    return has


def run_gate(mp4: Path, has_det: np.ndarray, cfg: dict) -> np.ndarray:
    vc = cfg["virat"]
    if vc["asmag_src"] not in sys.path:
        sys.path.insert(0, vc["asmag_src"])
    from gating.gates import ASMAGPlusGate  # noqa: E402  (read-only third-party source)
    with open(vc["asmag_config"], encoding="utf-8") as fh:
        gate_cfg = dict(yaml.safe_load(fh).get("p4_asmag_plus", {}))
    gate = ASMAGPlusGate(gate_cfg)
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    score = np.zeros(n, np.float32)
    valid = np.zeros(n, np.int8)
    prev = None
    rw = int(vc["resize_width"])
    for t in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (rw, max(1, int(h * rw / w))), interpolation=cv2.INTER_LINEAR)
        _, _, info = gate.process(bgr, prev)
        gate.update_after_detection(bool(has_det[t]) if t < len(has_det) else False)
        score[t] = info["gate_score"]
        valid[t] = 1
        prev = bgr
    cap.release()
    return score, valid


def event_count(e: np.ndarray) -> int:
    return int((np.diff(np.concatenate([[0], e])) == 1).sum())


def main() -> None:
    cfg = load_config()
    vdir, adir, ddir = virat_paths(cfg)
    ddir.mkdir(parents=True, exist_ok=True)
    disc = set(cfg["virat"]["disc_activities"])
    rows = []
    for mp4 in sorted(vdir.glob("*.mp4")):
        clip = mp4.stem
        act, geom = find_annotation(adir, clip, "activities"), find_annotation(adir, clip, "geom")
        cap = cv2.VideoCapture(str(mp4))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if act is None or geom is None:
            rows.append(dict(clip=clip, n_frames=n, fps=round(fps, 3), annotated=0))
            print(f"{clip}: no DIVA annotation -> excluded", flush=True)
            continue
        free_gb = psutil.virtual_memory().available / 2**30
        if free_gb < cfg["virat"]["min_free_ram_gb"]:
            sys.exit(f"free RAM {free_gb:.1f} GB < {cfg['virat']['min_free_ram_gb']} GB — stopping")
        e_disc, e_veh, names = activity_masks(act, n, disc)
        has_det = detection_mask(geom, n)
        out = ddir / f"{clip}.npz"
        if out.exists():
            z = np.load(out)
            score, valid = z["score"], z["valid"]
        else:
            t0 = time.time()
            score, valid = run_gate(mp4, has_det, cfg)
            np.savez_compressed(out, score=score, valid=valid, e_disc=e_disc, e_veh=e_veh, has_det=has_det)
            print(f"{clip}: {n} frames in {time.time() - t0:.0f}s", flush=True)
        rows.append(dict(clip=clip, n_frames=n, fps=round(fps, 3), annotated=1, split=act.parent.name,
                         n_activities=len(names), pi_disc=round(float(e_disc.mean()), 4),
                         n_events_disc=event_count(e_disc), pi_veh=round(float(e_veh.mean()), 4),
                         n_events_veh=event_count(e_veh), det_frame_rate=round(float(has_det.mean()), 4),
                         score_mean=round(float(score.mean()), 4),
                         score_in_disc=round(float(score[e_disc == 1].mean()) if e_disc.any() else np.nan, 4),
                         score_out_disc=round(float(score[e_disc == 0].mean()) if (e_disc == 0).any() else np.nan, 4)))
    inv = pd.DataFrame(rows)
    inv.to_csv(results_dir(cfg) / "P1_virat_inventory.csv", index=False)
    print(inv.to_string())


if __name__ == "__main__":
    main()
