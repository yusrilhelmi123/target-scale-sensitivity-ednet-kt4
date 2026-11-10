# Sensitivity results

`multisplit_seed2026_raw.jsonl` contains the exact raw records that were present in the original experiment bundle for training seed 2026 on the three additional split seeds.

`multisplit_all_seeds_reported.csv` mirrors the final per-seed values reported in Supplementary Table S4 for four learner-disjoint splits and five training seeds per split. Values are stored at the precision reported in the supplementary material. `multisplit_summary_reported.csv` is computed from those rows.

`reward_select.jsonl` contains the reward-rescaling validation sweep used for the reward-scale sensitivity analysis.
