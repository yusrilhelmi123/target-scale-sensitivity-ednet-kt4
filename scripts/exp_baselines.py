#!/usr/bin/env python
"""Expanded baseline comparison.

Adds, on top of Offline Q / CQL / APTQ-inspired-scaled-RTG:
  * APTQ-ORIGINAL  -- target = quantile of the learned dataset-Q distribution
  * APTQ-reward    -- reward-quantile target (kept for continuity)
  * SPIBB          -- Pi_b safe policy improvement with baseline bootstrapping
  * BC-majority    -- fully evaluated behaviour-cloning policy
  * Random         -- calibration lower bound
  * Global-majority-- calibration reference

Uses a learner-disjoint train / validation / test split: all models are
compared on the untouched TEST learners. FQE value + model-based (empirical-MDP)
value + count-based support diagnostics are reported for every policy.

Usage:
  python exp_baselines.py --transitions-csv <file.csv> --out-csv results.csv \
      --seeds 2026 2027 2028
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

import common as C


def spibb_policy(counts, trans, rew_sum, sa_count, term, n_states, num_actions,
                 gamma, n_wedge=10, iters=60):
    """Pi_b-SPIBB (Laroche et al., 2019) on the empirical MDP.

    Bootstrapped set B = {(s,a): count(s,a) < n_wedge}. On bootstrapped actions
    the policy is pinned to the behaviour policy; the remaining probability mass
    is placed on the best non-bootstrapped action under the current Q. Returns a
    deterministic policy (argmax of the resulting action distribution).
    """
    behaviour = C.behaviour_policy_from_counts(counts)
    boot = counts < n_wedge                      # boolean (s,a)
    # start from behaviour policy distribution
    pi = behaviour.copy()
    Q = np.zeros((n_states, num_actions))
    for _ in range(iters):
        # policy evaluation of current stochastic pi on empirical MDP (one sweep of Q)
        Qnew = np.zeros((n_states, num_actions))
        for s in range(n_states):
            for a in range(num_actions):
                c = sa_count[s, a]
                if c == 0:
                    Qnew[s, a] = 0.0
                    continue
                r = rew_sum[s, a] / c
                nxt = trans[(s, a)]   # nonterminal successors only (v1.1)
                boot_val = 0.0
                for ns, cnt in nxt.items():
                    boot_val += (cnt / c) * float(np.sum(pi[ns] * Q[ns]))
                Qnew[s, a] = r + gamma * boot_val
        Q = Qnew
        # SPIBB policy improvement
        new_pi = np.zeros((n_states, num_actions))
        for s in range(n_states):
            mass_boot = float(np.sum(behaviour[s][boot[s]]))
            new_pi[s][boot[s]] = behaviour[s][boot[s]]
            free = np.where(~boot[s])[0]
            if len(free) > 0:
                best = free[int(np.argmax(Q[s][free]))]
                new_pi[s][best] += (1.0 - mass_boot)
            else:
                new_pi[s] = behaviour[s]
        pi = new_pi
    return np.argmax(pi, axis=1).astype(np.int64), Q


def run_seed(rows, seed, num_actions, gamma, n_wedge):
    train, val, test = C.split_by_user_three_way(rows, val_fraction=0.15, test_fraction=0.15, seed=seed)
    sidx = C.build_state_index(rows)
    counts = C.state_action_counts(train, sidx, num_actions)
    behaviour = C.behaviour_policy_from_counts(counts)

    # ---- learned Q policies ----
    policies = {}
    q_offline, _ = C.train_q(train, sidx, num_actions, algorithm="offline_q", seed=seed)
    q_cql, _ = C.train_q(train, sidx, num_actions, algorithm="cql", cql_alpha=0.20, seed=seed)
    q_norm, h_norm = C.train_q(train, sidx, num_actions, algorithm="aptq",
                               aptq_target_source="return_to_go_quantile",
                               aptq_target_transform="scale", aptq_target_scale=0.10,
                               aptq_target_quantile=0.80, seed=seed)
    q_rew, _ = C.train_q(train, sidx, num_actions, algorithm="aptq",
                         aptq_target_source="reward_quantile",
                         aptq_target_transform="identity", aptq_target_quantile=0.60, seed=seed)
    q_apt_orig, h_orig = C.train_q(train, sidx, num_actions, algorithm="aptq",
                                   aptq_target_source="dataset_q_quantile",
                                   aptq_target_quantile=0.80, seed=seed)
    policies["Offline Q"] = (C.greedy_policy(q_offline), q_offline)
    policies["CQL a=0.20"] = (C.greedy_policy(q_cql), q_cql)
    policies["APTQ-insp. norm-RTG"] = (C.greedy_policy(q_norm), q_norm)
    policies["APTQ-insp. reward-q"] = (C.greedy_policy(q_rew), q_rew)
    policies["APTQ-original dataset-Q"] = (C.greedy_policy(q_apt_orig), q_apt_orig)

    # ---- SPIBB ----
    trans, rew_sum, sa_count, term = C.build_empirical_mdp(train, sidx, num_actions)
    pi_spibb, q_spibb = spibb_policy(counts, trans, rew_sum, sa_count, term,
                                     len(sidx), num_actions, gamma, n_wedge=n_wedge)
    policies["SPIBB (n_wedge=%d)" % n_wedge] = (pi_spibb, q_spibb)

    # ---- calibration / imitation policies ----
    maj_action = int(np.argmax(counts.sum(axis=0)))
    pi_bc = np.array([int(np.argmax(counts[s])) if counts[s].sum() > 0 else maj_action
                      for s in range(len(sidx))], dtype=np.int64)
    policies["BC majority-action"] = (pi_bc, None)
    policies["Global-majority"] = (np.full(len(sidx), maj_action, dtype=np.int64), None)
    rng = np.random.default_rng(seed)
    policies["Random"] = (rng.integers(0, num_actions, size=len(sidx)).astype(np.int64), None)

    # ---- evaluate every policy on TEST learners ----
    out = []
    for name, (pi, q) in policies.items():
        m = C.policy_metrics(pi, test, sidx, counts, behaviour)
        m["policy"] = name
        m["seed"] = seed
        m["split"] = "test"
        m["bellman_mse"] = C.bellman_mse(q, test, sidx, gamma) if q is not None else ""
        m["fqe_value"] = C.fqe_value(pi, train, test, sidx, gamma)
        V = C.evaluate_policy_empirical_mdp(pi, trans, rew_sum, sa_count, term, len(sidx), gamma)
        # model-based value averaged over test-state visitation
        vv = [float(V[sidx[r["state"]]]) for r in test]
        m["empirical_mdp_value"] = float(np.mean(vv)) if vv else 0.0
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--num-actions", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--n-wedge", type=int, default=10)
    args = ap.parse_args()

    rows = C.read_transitions(args.transitions_csv)
    all_rows = []
    for seed in args.seeds:
        print(f"[baselines] seed {seed} ...", flush=True)
        all_rows.extend(run_seed(rows, seed, args.num_actions, args.gamma, args.n_wedge))
    C.write_rows(args.out_csv, all_rows)

    # aggregate mean/std across seeds
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        agg[r["policy"]].append(r)
    summary = []
    keys = ["action_match", "bellman_mse", "fqe_value", "empirical_mdp_value",
            "binary_in_support", "frac_sel_lt5", "behaviour_prob_mean", "tv_distance_mean"]
    for pol, rs in agg.items():
        row = {"policy": pol, "n_seeds": len(rs)}
        for k in keys:
            vals = [float(x[k]) for x in rs if x[k] != ""]
            row[k + "_mean"] = float(np.mean(vals)) if vals else ""
            row[k + "_std"] = float(np.std(vals)) if vals else ""
        summary.append(row)
    C.write_rows(args.out_csv.with_name(args.out_csv.stem + "_summary.csv"), summary)
    print(f"[baselines] wrote {args.out_csv} and *_summary.csv", flush=True)


if __name__ == "__main__":
    main()
