# E2 — Synthetic tabular MDP with known values: pre-specified design (written BEFORE any E2 run)

Written 2026-10-05, before executing e2_synthetic.py. Responds to Reviewer 2: "a controlled synthetic MDP with adjustable coverage could test the interpretation" (optimistic Offline-Q estimate; action agreement as safety).

## World generator (mirrors the scale of the EdNet proxy)
- 600 states, 7 actions, discount 0.95 (training), RTG discount 0.99, per-transition termination probability 0.02.
- Next-state kernel (given continue): with prob 0.5 draw from a global Zipf(1) popularity over states (skewed visitation), with prob 0.5 draw from 3 random local successors specific to (s,a).
- Reward: +1 with prob p(s,a)=sigmoid(u_s+v_(s,a)), else -1; u_s~N(0.2,0.5), v_(s,a)~N(0,0.6). True model is known, so exact policy values are computed by dynamic programming.
- Behaviour policy: pi_b(a|s) proportional to exp(beta*w(s,a)), w~N(0,1) independent of rewards. Coverage factor beta in {0.5 (broad), 1.5 (medium), 3.0 (narrow)}.
- Data: learner-disjoint train/val/test = 2600/560/560 episodes (~130k/28k/28k transitions). 20 independent worlds (MDP + data) per coverage level = 60 worlds.

## Learners (same pipeline as the manuscript)
Offline Q (alpha=0); CQL with alpha chosen from {0.01,0.02,0.05,0.1,0.2,0.5} by validation Bellman MSE; adaptive scaled-RTG learner (eta=0.02, alpha_0=0.20, quantile and scale chosen on validation Bellman MSE over scale {0.05,0.1,0.2,0.5} x quantile {0.6,0.8,0.9}); CQL matched to the coefficient actually used in the last adaptive epoch, alpha_(E-1). 20 epochs, lr 0.08.

## Estimators and truth (test-visitation weighting)
For each greedy policy: (i) the learned Q's own value max_a Q(s,a); (ii) FQE (60 epochs, lr 0.05) fitted on training data; (iii) empirical-MDP value (train-built model); (iv) TRUE value by exact DP under the true model. Bias = estimate - truth.

## Pre-specified hypotheses (reported whatever the outcome)
- H1: the Offline-Q learned value overestimates the true policy value in every coverage level (mean bias > 0), and the overestimation is largest at narrow coverage.
- H2: FQE and the empirical-MDP estimate are NOT reliable substitutes for truth: report their bias and their pairwise-ranking accuracy (Offline Q vs CQL vs adaptive) against the true ranking.
- H3: action agreement with the logged action is not a safe proxy for true value: report the Spearman correlation between action match and the true value improvement over the behaviour policy, pooled within coverage level.
- H4: the adaptive learner and its alpha_(E-1)-matched CQL have the same true value up to +/-0.10 (the manuscript's FQE margin), paired by world; report the paired difference with a bootstrap interval over worlds.
No hypothesis, margin or metric above is altered after the results are seen; any later change is reported with its reason.
