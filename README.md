# Reproducibility archive: Target-Scale Sensitivity in Adaptive Pessimism (EdNet-KT4)

This archive accompanies the manuscript **"Target-Scale Sensitivity in Adaptive Pessimism for Tabular Offline Reinforcement Learning: An EdNet-KT4 Study."** It contains code, frozen protocol settings, primary clean-200k outputs, and sensitivity-analysis outputs. The EdNet source data are not redistributed and must be obtained separately from the authorised EdNet source under its applicable licence.

## Version 1.2 (2026-10-05): first-revision additions

- `revision_round1/` contains the analyses added in response to the first review round: `R1.1_state_audit` (no-gap robustness), `R1.2_rematch` (CQL control matched to alpha_(E-1), disagreement and value gaps), `R1.3_spibb` (deterministic vs stochastic SPIBB evaluation and threshold grid), `R1.4_bootstrap` (pooled learner-only, seed-t and two-way intervals), `E1_factorial` (preregistered 36-configuration factorial study) and `E2_synthetic` (preregistered synthetic-MDP study with known values) and `R2.3_independent_prereg` (pre-specified design, hash and timestamp for an independent evaluation on EdNet learners not used in this study; the evaluation itself has not been run). Preregistration files carry SHA-256 digests and timestamps recorded before the runs.
- `ARTIFACT_MANIFEST.md` maps every manuscript and supplementary table to its files and scripts and lists the exact binning/filtering configuration, split identifiers, and dataset checksum.
- Results in `results_clean_200k/` are unchanged from v1.1.

## Version 1.1 (2026-09-27): changes relative to v1.0

- **Empirical-MDP kernel correction.** `common.build_empirical_mdp` now accumulates successor counts from non-terminal transitions only, and `common.evaluate_policy_empirical_mdp` / `exp_baselines.spibb_policy` bootstrap with `P_c(s'|s,a) = N_cont(s,a,s')/N(s,a)` (which already sums to `1 - d(s,a)`). v1.0 normalised the kernel over all transitions and additionally multiplied by `(1 - d)`, double-counting termination for state-action pairs that end a trajectory only occasionally. The correction changes empirical-MDP values by at most 0.2, changes the SPIBB policy slightly (test FQE 6.380 -> 6.576), and leaves every policy ranking and the validation-selected SPIBB threshold (`n_wedge = 2`) unchanged. `results_clean_200k/` holds the corrected per-seed and summary metrics; all learned Q-tables and every non-empirical-MDP metric of the other policies are unchanged. `results_v11/v10_reproduction_check.csv` confirms that the v1.0 formula reproduces the archived v1.0 values exactly (max abs difference < 2e-15) before the correction is applied.
- **Additional controls** (`scripts/exp_v11_controls.py`, outputs in `results_v11/`):
  - `gamma.jsonl`: RTG-discount ablation (RTG recomputed with `gamma_rtg = 0.95`, equal to the training discount; target scale/quantile re-selected on validation on a grid extended with an unscaled target; five seeds with final-alpha-matched CQL). Main-text Section 4.7 / Table 10; Supplementary Table S11.
  - `reward.jsonl`: reward-unit controls (multipliers 0.5/1/2; five seeds; final-alpha-matched CQL, fixed CQL(0.05), CQL with alpha re-selected per multiplier). Main-text Table 10; Supplementary Table S12.
  - `long_horizon.jsonl` (`scripts/exp_v11_long_horizon.py`): 100-epoch coefficient and mean dataset-Q trajectories for the scaled-RTG, reward-quantile and moving dataset-Q targets (five seeds). Supplementary Table S13.
  - `near_tie_check.json` (`scripts/near_tie_check.py`): diagnostic for the large single-run gaps in the multi-split check (split 5678 seed 2026; split 9012 seed 2030): states where the greedy actions of APTQ and matched CQL differ, their share of test transitions, and top-two Q-value gaps; `near_tie_decomposition.json` gives the per-state contribution to each run's action-match gap. Main-text Section 4.7.
  - `tost.json`: post hoc margin-based equivalence analysis (90% t intervals; margins ±0.02 action match, ±0.10 FQE). Supplementary Table S10.
- **Acceleration.** `scripts/fastrl.py` is a Numba-compiled drop-in for `common.train_q` and `common.fqe_fit` used by the v1.1 control experiments; `scripts/check_fastrl.py` verifies that it reproduces the archived v1.0 Q-tables (max abs difference ~1e-16) and FQE value exactly. The v1.0 scripts do not require Numba.
- `scripts/empirical_mdp_v11.py` re-evaluates all archived policies with the corrected kernel (using the per-seed Q-tables now included as `results_clean_200k/seed_*_q_tables.npz`; point its second argument at a folder holding the v1.0 `seed_*_metrics.csv` files from archive v1.0 together with these Q-tables) and re-runs SPIBB selection and evaluation; `scripts/legacy/common_v10.py` is the v1.0 helper module used for the reproduction check.

### Reproduce the v1.1 additions

```bash
python scripts/check_fastrl.py DATA.csv results_clean_200k/     # uses the archived per-seed Q-tables (seed_*_q_tables.npz)
python scripts/exp_v11_controls.py DATA.csv out_v11/ gamma
python scripts/exp_v11_controls.py DATA.csv out_v11/ reward
python scripts/exp_v11_long_horizon.py DATA.csv out_v11/
python scripts/near_tie_check.py DATA.csv out_v11/near_tie_check.json
```

## Environment

- Python >= 3.10
- Install dependencies:

```bash
pip install -r requirements.txt
```

The supplied environment uses `numpy==2.2.6` and `matplotlib==3.10.9`; the v1.1 scripts additionally use `numba` (tested with 0.67) and `pandas`.

## Final protocol

The final protocol is recorded in `configs/final_protocol.json`:

- 200,000 ordered transitions from 2,665 learners
- 626 tabular states and 7 question-part actions
- learner-disjoint 70/15/15 train/validation/test split
- primary split seed: `20260728`
- selection-training seed: `2026`
- final training seeds: `2026–2030`
- Q-learning rate: `0.08`
- discount factor: `0.95`
- RTG discount: `0.99`
- scaled-RTG target: scale `0.10`, quantile `0.90`
- APTQ-inspired initial alpha `0.20`, adaptation learning rate `0.02`

The SHA-256 fingerprint of the analysed transition file is:

```text
9a11461d6db774fc43ee784650a4abd77af45ec110bae553d3cc16919f4248f9
```

## Reconstruct the 200k transition file

Obtain `EdNet-KT4.zip` and `EdNet-Contents.zip` from the authorised EdNet source, then run:

```bash
python scripts/preprocess_ednet_kt4.py \
  --kt4-zip /path/to/EdNet-KT4.zip \
  --contents-zip /path/to/EdNet-Contents.zip \
  --out-csv DATA.csv \
  --metadata-json DATA.metadata.json \
  --max-users 4000 \
  --max-transitions 200000 \
  --min-transitions-per-user 2 \
  --reward-correct 1.0 \
  --reward-incorrect -1.0 \
  --reward-mode correctness \
  --rtg-gamma 0.99 \
  --action-space question_part \
  --state-variant basic
```

Verify `DATA.csv` against the SHA-256 value above. `results_clean_200k/split_manifest.json` records split counts and learner-list hashes without redistributing raw learner identifiers.

## Archive layout

- `configs/final_protocol.json`: frozen final protocol
- `scripts/preprocess_ednet_kt4.py`: transition construction
- `scripts/clean_200k.py`: clean experiment, FQE, aggregation, bootstrap diagnostics
- `scripts/common.py`: tabular training/evaluation helpers
- `scripts/exp_baselines.py`: SPIBB and extended baselines
- `scripts/exp_sensitivity.py`: split and reward-scale sensitivity driver
- `scripts/run_multisplit_pair.py`: four splits × five seeds, APTQ-inspired scaled RTG followed by alpha-matched CQL
- `scripts/cql_fine_grid.py`: five-seed CQL fine-grid re-check
- `scripts/make_figures.py`: Figures 2, 3, and S1
- `results_clean_200k/`: primary selection, per-seed/aggregate metrics, paired comparisons, histories, fine-grid and action-distribution tables
- `results_sensitivity/`: learner-split and reward-rescaling outputs

The internal policy identifier `aptq_norm_rtg` is retained in legacy result files and code paths for backward compatibility. Human-readable labels use **scaled RTG**, which is the terminology used in the final manuscript.

## Reproduce primary clean-200k results

```bash
python scripts/clean_200k.py select --transitions-csv DATA.csv --out-dir out/
for S in 2026 2027 2028 2029 2030; do
  python scripts/clean_200k.py evaluate \
    --transitions-csv DATA.csv --out-dir out/ \
    --selection-json out/selection.json \
    --train-seed $S --fqe-epochs 60 --n-boot 1000
  python scripts/clean_200k.py complete-fqe \
    --transitions-csv DATA.csv --out-dir out/ \
    --selection-json out/selection.json \
    --train-seed $S --fqe-epochs 60
done
python scripts/clean_200k.py aggregate --out-dir out/
```

## Reproduce the CQL fine-grid check

```bash
python scripts/cql_fine_grid.py \
  --transitions-csv DATA.csv \
  --out-csv out/cql_fine_grid.csv
```

The archived `results_clean_200k/cql_fine_grid_reported.csv` mirrors the final supplementary table at the precision reported there.

## Reproduce four-split × five-seed sensitivity

Use an empty output directory:

```bash
python scripts/run_multisplit_pair.py \
  --transitions-csv DATA.csv \
  --out-dir out_sensitivity/
```

This runs split seeds `20260728`, `1234`, `5678`, and `9012`, with training seeds `2026–2030`. For each split/seed pair it trains the APTQ-inspired scaled-RTG policy and then fixed CQL with alpha matched to the adaptive run's final coefficient.

`results_sensitivity/multisplit_all_seeds_reported.csv` mirrors the final Supplementary Table S4 at the precision reported in the manuscript. The exact raw records originally bundled for training seed 2026 are retained separately as `multisplit_seed2026_raw.jsonl`.

## Reproduce reward-scale sensitivity

```bash
for M in 0.5 1.0 2.0; do
  for SC in 0.05 0.10 0.20; do
    python scripts/exp_sensitivity.py rs \
      --transitions-csv DATA.csv \
      --out-dir out_reward_scale/ \
      --mult $M --scale $SC --quant 0.90 --seed 2026 --split 20260728
  done
done
```

## Reproduce bundled figures without external data

```bash
python scripts/make_figures.py \
  --results-dir results_clean_200k \
  --out-dir figures
```

This writes `figure2_tradeoff.png`, `figure3_target.png`, and `figureS1_trajectories.png`.

## Result-file mapping

- Main Table 3: `summary_metrics.csv`
- Main Table 4 and Supplementary Table S2: `summary_paired.csv`, `all_seed_paired.csv`
- Main Table 5 / calibration baselines: `summary_metrics.csv` plus the baseline outputs produced by `exp_baselines.py`
- Main Table 6: support-count and TV-distance fields in `summary_metrics.csv`
- Main Table 7 / Supplementary Table S1: `selection_grid.csv`; fine-grid re-check: `cql_fine_grid_reported.csv`
- Main Table 8: FQE and empirical-MDP fields in `summary_metrics.csv`
- Main Table 9 / Supplementary Table S4: `results_sensitivity/multisplit_all_seeds_reported.csv`
- Main Table 10 / Supplementary Table S5: `results_sensitivity/reward_select.jsonl`
- Supplementary Table S9: `action_distribution_reported.csv`
- Supplementary Table S10: `results_v11/tost.json`
- Supplementary Table S13: `results_v11/long_horizon.jsonl`
- Supplementary Table S11 and Table 10 (RTG-discount row): `results_v11/gamma.jsonl`
- Supplementary Table S12 and Table 10 (reward-unit rows): `results_v11/reward.jsonl`
- Empirical-MDP / SPIBB correction: `results_v11/spibb_selection_v11.csv`, `results_v11/v10_reproduction_check.csv`

## Statistical interpretation

Seed-level intervals quantify training-seed stochasticity on the primary split. The learner bootstrap is a single-level learner-resampling diagnostic conditional on trained policy instances; it is not a hierarchical learner-and-training-seed uncertainty interval. The four-split × five-seed analysis is reported separately to characterise sensitivity to learner partitioning.

## Citation

Citation metadata are provided in `CITATION.cff`. Add the Zenodo DOI to the Zenodo record after deposition and, if desired, update the citation metadata in the repository release.
