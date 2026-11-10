# Artifact-to-table manifest (archive v1.2, first revision, 2026-10-05)

Dataset (not redistributed; reconstruct with `scripts/preprocess_ednet_kt4.py`, see README):
`ednet_kt4_step6_large200k_part_basic_correctness_offline_rl.csv`, 200,000 transitions, 2,665 learners, 626 states, 7 actions,
SHA-256 `9a11461d6db774fc43ee784650a4abd77af45ec110bae553d3cc16919f4248f9` (also in `configs/final_protocol.json` and `results_clean_200k/split_manifest.json`).

Split: learner-disjoint 70/15/15, split seed 20260728 (`results_clean_200k/split_manifest.json`: counts and learner-list hashes; train 1,867 learners / 132,756 transitions, validation 399 / 38,634, test 399 / 28,610). Additional splits 1234, 5678, 9012 in `results_sensitivity/`. Training seeds 2026-2030; selection-training seed 2026.

Binning / filtering configuration (implemented in `scripts/preprocess_ednet_kt4.py`, documented in Supplementary Table S14):
events = action_type "respond", item id begins with "q", non-empty answer, question in metadata with part 1-7, parseable timestamp; sorted by timestamp (ties keep file order);
attempt bins 0 | 1-4 | 5-14 | 15-49 | >=50; accuracy bin 0 if no earlier response else min(4, 1+floor(4*accuracy)); gap bins first | <1 min | 1-<10 min | 10 min-<24 h | >=24 h;
attempts, accuracy, previous part and previous correctness use responses before the current one only; initial state a0_c0_p0_r2_g0; learners taken in archive order from the first 4,000 files (>=2 transitions) until the 200,000-transition cap (learner 4809 truncated at 114 transitions; its return-to-go uses the full trajectory).

| Manuscript / supplement item | Files (relative to archive root) | Code |
|---|---|---|
| Table 3 (main results), Tables S7, S8 | `results_clean_200k/summary_metrics.csv`, `all_seed_metrics.csv`, `seed_*_metrics.csv`; strictly matched alpha_(E-1) row: `revision_round1/R1.2_rematch/r12_per_seed.csv`, `r12_results.json` | `scripts/clean_200k.py`; `revision_round1/R1.2_rematch/r12_rematch.py` |
| Table 4 (paired differences), Table S2 | `results_clean_200k/summary_paired.csv`, `all_seed_paired.csv`, `seed_*_paired.csv`; Table S16 and S2 basis: `revision_round1/R1.4_bootstrap/r14_pooled_intervals.csv/.json`, `r14_current_tableS2_basis.csv` | `scripts/clean_200k.py`; `revision_round1/R1.4_bootstrap/r14_bootstrap.py` |
| Table 5 (extended baselines), Table S17 (SPIBB) | `results_clean_200k/summary_metrics.csv`; `results_v11/spibb_selection_v11.csv`; `revision_round1/R1.3_spibb/r13_spibb_grid.csv`, `r13_spibb.json` | `scripts/exp_baselines.py`, `scripts/empirical_mdp_v11.py`; `revision_round1/R1.3_spibb/r13_spibb.py` |
| Table S1 (selection grids), S3 | `results_clean_200k/selection_grid.csv`, `selection.json`, `cql_fine_grid_reported.csv` | `scripts/clean_200k.py select`, `scripts/cql_fine_grid.py` |
| Figures 1-4 (revision, IJIES format: single panels, no panel labels) | `results_clean_200k/summary_metrics.csv`, `seed_*_histories.csv`, `selection_grid.csv` | `scripts/make_figures_r1.py` (Figures 2, 3 and 4; Figure 1 is drawn in the same script) |
| Figure S1, epoch-level alpha and mean dataset Q | `results_clean_200k/seed_*_histories.csv` (epoch, alpha, mean_dataset_q, target_q, mean_abs_td_error per policy and seed) | `scripts/make_figures.py` |
| Learned policy tables (Q-tables) | `results_clean_200k/seed_*_q_tables.npz` (10 policies per seed); alpha_(E-1) control: `revision_round1/R1.2_rematch/q_cql_alphaEm1_seed*.npy` | `scripts/clean_200k.py`; `r12_rematch.py` |
| Table 9, Table S4 (four splits) | `results_sensitivity/multisplit_*.csv/.jsonl` | `scripts/run_multisplit_pair.py` |
| Table 10, Tables S5, S12 (reward unit) | `results_v11/reward.jsonl`, `results_sensitivity/reward_select.jsonl` | `scripts/exp_v11_controls.py`, `scripts/exp_sensitivity.py` |
| Table S11 (RTG discount) | `results_v11/gamma.jsonl` | `scripts/exp_v11_controls.py` |
| Table S13 (100 epochs) | `results_v11/long_horizon.jsonl` | `scripts/exp_v11_long_horizon.py` |
| Table S9 (action distributions) | `results_clean_200k/action_distribution_reported.csv` | `scripts/clean_200k.py` |
| Table S10 (TOST), near-tie decomposition | `results_v11/tost.json`, `near_tie_check.json`, `near_tie_decomposition.json` | `scripts/near_tie_check.py` |
| Table S14 (state audit, no-gap check) | `revision_round1/R1.1_state_audit/r11_nogap.json` | `r11_nogap.py` |
| Table S15 (alpha_(E-1) per seed, disagreement, value gaps) | `revision_round1/R1.2_rematch/r12_per_seed.csv`, `r12_results.json`, `r12_table_rows.json` | `r12_rematch.py`, `r12_tables.py` |
| Table 11, Table S18 (factorial study) | `revision_round1/E1_factorial/e1_cell_summary.csv`, `e1_runs_flat.csv`, `e1_runs.json`, `e1_cql_best.json`, `e1_cql_budget_grid.json`, `E1_preregistration.md` + `.sha256` + `prereg_timestamp.txt` | `e1_factorial.py` |
| Table 12, Table S19 (synthetic MDP) | `revision_round1/E2_synthetic/e2_worlds_beta*.jsonl`, `e2_worlds_flat.csv`, `e2_analysis_output.txt`, `E2_preregistration.md` + `.sha256` + `prereg_timestamp.txt` | `e2_synthetic.py`, `e2_analyze.py` |

| Pre-specified independent EdNet evaluation (Response R2.3; design only, not yet run) | `revision_round1/R2.3_independent_prereg/EdNet_independent_preregistration.md` + `.sha256` + `prereg_timestamp.txt`, `excluded_learner_ids.txt` (the 2,665 learners of the analysed subset) | not applicable (no run yet) |

Notes: scripts under `revision_round1/` take the data CSV and archive path as command-line arguments (see the header of each script) and were run with the v1.1 `scripts/fastrl.py` kernel; `CHECKSUMS.sha256` at the root lists every file. The internal identifier `aptq_norm_rtg` denotes the scaled-RTG adaptive policy.
