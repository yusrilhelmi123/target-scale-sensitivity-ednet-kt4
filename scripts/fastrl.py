"""Numba-accelerated drop-in versions of common.train_q and common.fqe_fit.

Same update rules, same epoch-wise shuffling (numpy default_rng(seed + epoch)
for training, default_rng(seed + ep) for FQE); only the inner per-transition
loop is compiled. Used for the v1.1 additional experiments (RTG-discount
ablation and reward-scale controls). Equivalence with common.py is checked in
check_fastrl.py.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

import common as C


@njit(cache=True)
def _epoch(q, S, NS, A, R, D, order, lr, gamma, alpha, conservative):
    abs_sum = 0.0
    na = q.shape[1]
    for k in range(order.shape[0]):
        i = order[k]
        s = S[i]; a = A[i]
        if D[i]:
            target = R[i]
        else:
            m = q[NS[i], 0]
            for j in range(1, na):
                if q[NS[i], j] > m:
                    m = q[NS[i], j]
            target = R[i] + gamma * m
        td = target - q[s, a]
        q[s, a] += lr * td
        abs_sum += abs(td)
        if conservative and alpha > 0:
            m = q[s, 0]
            for j in range(1, na):
                if q[s, j] > m:
                    m = q[s, j]
            tot = 0.0
            e = np.empty(na)
            for j in range(na):
                e[j] = np.exp(q[s, j] - m)
                tot += e[j]
            for j in range(na):
                q[s, j] -= lr * alpha * (e[j] / tot)
            q[s, a] += lr * alpha
    return abs_sum


def train_q(train_rows, state_index, num_actions, *, algorithm="cql", epochs=20, lr=0.08,
            gamma=0.95, cql_alpha=0.20, aptq_initial_alpha=0.20, aptq_alpha_lr=0.02,
            aptq_target_source="return_to_go_quantile", aptq_target_transform="scale",
            aptq_target_scale=0.10, aptq_target_offset=0.0, aptq_target_quantile=0.80,
            aptq_fixed_target=None, seed=2026, rtg_key="return_to_go", arrays=None):
    q = np.zeros((len(state_index), num_actions), dtype=np.float64)
    alpha = {"offline_q": 0.0, "cql": cql_alpha, "aptq": aptq_initial_alpha}[algorithm]
    static_target = None
    if algorithm == "aptq" and aptq_target_source in ("reward_quantile", "return_to_go_quantile"):
        if aptq_target_source == "reward_quantile":
            vals = np.array([r["reward"] for r in train_rows], dtype=np.float64)
        else:
            vals = np.array([r[rtg_key] for r in train_rows], dtype=np.float64)
        vals = C.transform_values(vals, aptq_target_transform, aptq_target_scale, aptq_target_offset)
        static_target = float(np.quantile(vals, aptq_target_quantile))
    elif algorithm == "aptq" and aptq_target_source == "fixed_value":
        static_target = float(aptq_fixed_target)
    arr = arrays if arrays is not None else C.to_arrays(train_rows, state_index)
    S, NS, A, R, D = arr["s"], arr["ns"], arr["a"], arr["r"], arr["done"]
    n = len(S)
    order = np.arange(n)
    conservative = algorithm in ("cql", "aptq")
    history = []
    for epoch in range(1, epochs + 1):
        rng2 = np.random.default_rng(seed + epoch)
        rng2.shuffle(order)
        abs_sum = _epoch(q, S, NS, A, R, D, order, lr, gamma, float(alpha), conservative)
        data_q = q[S, A]
        mean_dataset_q = float(data_q.mean()) if n else 0.0
        if algorithm == "aptq":
            if aptq_target_source == "dataset_q_quantile":
                target_q = float(np.quantile(data_q, aptq_target_quantile)) if len(data_q) else 0.0
            else:
                target_q = static_target
            alpha = max(0.0, alpha + aptq_alpha_lr * (mean_dataset_q - target_q))
        else:
            target_q = ""
        history.append({"epoch": epoch, "algorithm": algorithm, "alpha": alpha, "target_q": target_q,
                        "mean_dataset_q": mean_dataset_q,
                        "mean_abs_td_error": (abs_sum / n) if n else 0.0})
    return q, history


@njit(cache=True)
def _fqe_epoch(q_eval, S, NS, A, R, D, pol, order, lr, gamma):
    for k in range(order.shape[0]):
        i = order[k]
        tgt = R[i]
        if not D[i]:
            tgt += gamma * q_eval[NS[i], pol[NS[i]]]
        q_eval[S[i], A[i]] += lr * (tgt - q_eval[S[i], A[i]])


def fqe_fit(policy, train_rows, state_index, gamma, lr=0.05, epochs=60, seed=2026, num_actions=7,
            arrays=None):
    q_eval = np.zeros((len(state_index), num_actions), dtype=np.float64)
    arr = arrays if arrays is not None else C.to_arrays(train_rows, state_index)
    S, NS, A, R, D = arr["s"], arr["ns"], arr["a"], arr["r"], arr["done"]
    pol = np.asarray(policy, dtype=np.int64)
    order = np.arange(len(S))
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(order)
        _fqe_epoch(q_eval, S, NS, A, R, D, pol, order, lr, gamma)
    return q_eval


def fqe_value(policy, train_rows, eval_rows, state_index, gamma, lr=0.05, epochs=60, seed=2026,
              num_actions=7, arrays=None):
    q_eval = fqe_fit(policy, train_rows, state_index, gamma, lr, epochs, seed, num_actions, arrays)
    vals = C.fqe_state_values(q_eval, eval_rows, policy, state_index)
    return float(vals.mean()) if len(vals) else 0.0
