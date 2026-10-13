# Reproduction check (2026-10-14)

**Scope.** The first block of the README "Reproduce" section, run from a fresh `git clone` of commit `72e2272`
(Windows 11, Python 3.11.9, package versions of `requirements.txt`, laptop CPU, no GPU):

1. delete `results/numbers.json`, `results/real_trace/real_numbers.json`, `results/real_trace/calib.json`,
   `results/real_trace/aligned_*.csv` and `results/real_trace/P3_real_*.csv`;
2. `python real_trace/align.py` (re-reads the raw device dump; checks every schedule on the device against the schedule
   recomputed from the corpus, SHA-256 and element-wise);
3. `python real_trace/attack_real.py` (about 13 min);
4. `python p1_report.py` (all paper numbers and figures from the result tables, merged with `real.*`).

`CFS_PROCESSED_DIR` pointed to the C121 inputs of the IoT-J pipeline (see README, "Data"); `THS_DATASETS` was unset,
so the VIRAT derived files of this repository were used.

**Result.**

| Check | Outcome |
|---|---|
| `results/numbers.json`, key by key | **410 / 410 keys identical** (0 differing, 0 missing, 0 extra) |
| `results/numbers.json`, bytes | identical, SHA-256 `36a9c2d24ef2bd470e451a2cdbe2ba4b9f1524670b2e845cb3e91fb68dcd31c6` |
| `results/real_trace/real_numbers.json` | 164 / 164 keys identical |
| every other tracked file after the run (`git status`) | unchanged: regenerated tables, aligned traces, calibration and figures are byte-identical |

**Not re-run here.** The full experiment grid (`attacks.py` over all corpora, `verify_bounds.py`, `rho_sweep.py`,
SYN v2) takes hours; its outputs are the committed tables in `results/` from which step 4 regenerates every number.
