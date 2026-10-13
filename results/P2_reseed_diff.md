# P2 T17 — re-seed diff (09-10-2026)

**Change.** `code/verify_bounds.py` seeded two RNG streams with `hash(lab) % 2**31` (Thm 2 gate randomness, line ~109;
Thm 3 gate + iso-periodic phase, line ~308). Python salts `str.__hash__` per process, so those streams changed on every
run unless `PYTHONHASHSEED` was fixed (P1b ran with `PYTHONHASHSEED=0`; P1 ran unsalted, i.e. irreproducible).
Now every stream is `default_rng([SEED, ...])` with `SEED` from `--seed` (default 42) and gate labels mapped by
`zlib.crc32` (same scheme as `attacks.py`). No other file used `hash()` (`attacks.py` crc32; `rho_sweep.py`,
`syn_v2.py` explicit seeds). Re-run: `python verify_bounds.py thm2 thm1 thm3 thm2v2 --seed 42` (3 min 38 s, PYTHONHASHSEED unset).

| Table | Rows changed | Max |Δ| | Conclusion changed? |
|---|---|---|---|
| `P1_bounds_thm1.csv`, `_per_video.csv` | 0 | 0 | no (never used hash) |
| `P1_bounds_thm2.csv` | 50 (45 G8 CA+refresh, 5 G4 periodic — random phases) | p̂₁ 0,038; I 0,0076 bit; f 0,011 | **no**: 35/35 certified rows satisfy the bound, min slack 0, periodic P = 5 I = 0 |
| `P1_bounds_thm2_cor21a.csv` | 0 | 0 | no |
| `P1b_bounds_thm2_synv2.csv` | G8/G4 rows | p̂₁ 0,039 | **no**: 35/35 |
| `P1_bounds_thm3.csv` | all (gate RR draws + iso-periodic phase) | mean latency 0,47 frame; short-miss 0,11; ρ 0,005 | **no** (see below) |

**Thm 3 seed sensitivity** (seeds 42–46; iso-periodic comparator uses one random phase per clip, hence the spread):

| Corpus | RR (24) beats periodic, latency | RR beats periodic, short-miss | G1 τ 0,55 latency vs periodic | G3 latency vs periodic |
|---|---|---|---|---|
| C121 | 0 all seeds | 0 all seeds | 0,466 vs 0,196–0,213 | 0,366 vs 0,139–0,153 |
| VIRAT | 0–2 | — (no D < L) | 0,827 vs 0,373–0,600 | 0,640 vs 0,387–0,493 |
| SYN π 0,05 | 0–1 | 20–24 | 0,593 vs 0,445–0,508 | 0,490 vs 0,393–0,451 |
| SYN π 0,1 | 0–1 | 22–23 | 0,523 vs 0,467–0,508 | 0,531 vs 0,402–0,442 |

CP-B1 §3 numbers (old, unsalted run) are kept as the historical record in `CP_B1_39_20261105.md`; the manuscript uses the
seed-42 numbers from `numbers.json`. The CP-B1 statement "periodic at equal ρ dominates G1/G3 on latency and short-miss
on C121/VIRAT" holds for every seed. VIRAT RR "0/24 latency wins" becomes "0–2/24" — reported as such.
