# E1 — Factorial stability experiment: pre-specified design (written BEFORE any E1 run)

Written: 2026-10-05, before executing e1_factorial.py. Responds to Reviewer 2 (stability evidence varies target and horizon but fixes adaptation rate and initial coefficient).

## Design
- Data/split/protocol: unchanged (clean 200k transitions, split seed 20260728, 20 epochs, lr 0.08, gamma 0.95, five training seeds 2026-2030).
- Factors for the adaptive learner (scaled-RTG target, quantile 0.90 fixed):
  - target scale in {0.05, 0.10, 0.20, 0.50}
  - adaptation rate eta in {0.005, 0.02, 0.08}   (0.25x, 1x, 4x of the main setting 0.02)
  - initial coefficient alpha_0 in {0.05, 0.20, 0.50} (0.20 is the main setting)
  => 36 configurations x 5 seeds = 180 adaptive runs. All cells are reported, including failures.
- Fixed-CQL comparator with a comparable tuning budget: 36 alpha values, geometric grid 0.01 to 1.0; for each seed, the alpha with the lowest validation Bellman MSE is the "best CQL" (36 configurations, equal to the adaptive budget).
- Matched control for every adaptive run: CQL trained with the coefficient actually used in the last adaptive epoch, alpha_(E-1) (see R1.2).

## Run-level stability criterion (fixed in advance)
A run is STABLE iff all hold:
1. alpha after epoch 20 is at least 0.01 (no collapse to zero);
2. alpha never exceeds 1.0 during the 20 epochs (no inflation);
3. validation Bellman MSE of the adaptive Q-table is at most 1.10 x the validation MSE of the best fixed CQL for the same seed.
A cell (configuration) is STABLE iff at least 4 of its 5 seeds are stable; it is MIXED with 2-3 stable seeds; otherwise FAILED.

## What is reported
- Per-cell stable-seed counts and the reason for each failure (collapse / inflation / MSE gap), for all 36 cells.
- Final alpha range per cell and seed-to-seed spread.
- For stable runs: test action-match and FQE differences between the adaptive policy and its alpha_(E-1)-matched CQL (descriptive, exploratory; no new equivalence claim).
- Selection of any configuration uses validation metrics only. Test metrics are descriptive.
No criterion above is changed after the results are seen; if changed, the change and its reason are stated.
