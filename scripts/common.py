#!/usr/bin/env python
"""Shared utilities for the EdNet-KT4 tabular offline-RL experiments.

Supports additional sensitivity and robustness runs on the existing
EdNet-KT4 transition CSVs (same schema as train_tabular_offline_rl.py):

    user_id, timestep, timestamp, state, action(1-indexed), reward,
    next_state, done, question_id, correct, source, platform,
    source_file, return_to_go

Everything here is pure numpy + stdlib so it runs anywhere the original
pipeline runs. Import these helpers from the exp_*.py scripts.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# IO and splits
# --------------------------------------------------------------------------
def read_transitions(path: Path) -> List[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["action"] = int(row["action"]) - 1          # -> 0-indexed
        row["reward"] = float(row["reward"])
        row["done"] = int(row["done"])
        row["return_to_go"] = float(row["return_to_go"])
        row["timestep"] = int(row["timestep"])
    return rows


def split_by_user_three_way(
    rows: List[dict], val_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Learner-disjoint train / validation / test split.

    Selection uses validation learners; the final number is reported on
    the untouched test learners. No learner appears in more than one split.
    """
    users = sorted({row["user_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(users)
    n = len(users)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test_users = set(users[:n_test])
    val_users = set(users[n_test:n_test + n_val])
    train, val, test = [], [], []
    for row in rows:
        u = row["user_id"]
        if u in test_users:
            test.append(row)
        elif u in val_users:
            val.append(row)
        else:
            train.append(row)
    return train, val, test


def split_by_user(rows: List[dict], eval_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    users = sorted({row["user_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(users)
    k = max(1, int(len(users) * eval_fraction)) if len(users) > 1 else 0
    eval_users = set(users[:k])
    train = [r for r in rows if r["user_id"] not in eval_users]
    ev = [r for r in rows if r["user_id"] in eval_users]
    return train, ev


def build_state_index(rows: Iterable[dict]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for row in rows:
        for key in ("state", "next_state"):
            v = row[key]
            if v not in idx:
                idx[v] = len(idx)
    return idx


def to_arrays(rows: List[dict], state_index: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Vectorised view of a row list (built once, reused across epochs)."""
    n = len(rows)
    s = np.empty(n, dtype=np.int64); ns = np.empty(n, dtype=np.int64)
    a = np.empty(n, dtype=np.int64); r = np.empty(n, dtype=np.float64)
    done = np.empty(n, dtype=np.int64)
    for i, row in enumerate(rows):
        s[i] = state_index[row["state"]]; ns[i] = state_index[row["next_state"]]
        a[i] = row["action"]; r[i] = row["reward"]; done[i] = row["done"]
    return {"s": s, "ns": ns, "a": a, "r": r, "done": done}


# --------------------------------------------------------------------------
# Counts, behaviour policy, support diagnostics
# --------------------------------------------------------------------------
def state_action_counts(rows: Iterable[dict], state_index: Dict[str, int], num_actions: int) -> np.ndarray:
    counts = np.zeros((len(state_index), num_actions), dtype=np.int64)
    for row in rows:
        counts[state_index[row["state"]], row["action"]] += 1
    return counts


def behaviour_policy_from_counts(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    safe = np.where(totals == 0, 1, totals)
    pi = counts / safe
    # unseen states -> uniform
    pi[totals[:, 0] == 0] = 1.0 / counts.shape[1]
    return pi


def support_set_map(rows: Iterable[dict], state_index: Dict[str, int]) -> Dict[int, set]:
    support: Dict[int, set] = defaultdict(set)
    for row in rows:
        support[state_index[row["state"]]].add(row["action"])
    return support


# --------------------------------------------------------------------------
# Math helpers
# --------------------------------------------------------------------------
def empirical_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    s = np.sort(values)
    return np.searchsorted(s, values, side="right") / float(len(values))


def transform_values(values: np.ndarray, transform: str, scale: float, offset: float) -> np.ndarray:
    if transform in ("identity", "scale"):
        t = values
    elif transform == "minmax":
        lo, hi = (float(np.min(values)), float(np.max(values))) if len(values) else (0.0, 0.0)
        d = hi - lo
        t = np.zeros_like(values) if d == 0 else (values - lo) / d
    elif transform == "zscore":
        mu, sd = (float(np.mean(values)), float(np.std(values))) if len(values) else (0.0, 0.0)
        t = np.zeros_like(values) if sd == 0 else (values - mu) / sd
    elif transform == "rank":
        t = empirical_rank(values)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return t * scale + offset


# --------------------------------------------------------------------------
# Tabular training: offline_q / cql / aptq-inspired / aptq-original
# --------------------------------------------------------------------------
def _static_target_q(train_rows, transform, scale, offset, quantile, source) -> float:
    if source == "reward_quantile":
        vals = np.array([r["reward"] for r in train_rows], dtype=np.float64)
    elif source == "return_to_go_quantile":
        vals = np.array([r["return_to_go"] for r in train_rows], dtype=np.float64)
    else:
        return 0.0
    vals = transform_values(vals, transform, scale, offset)
    return float(np.quantile(vals, quantile))


def train_q(
    train_rows: List[dict],
    state_index: Dict[str, int],
    num_actions: int,
    *,
    algorithm: str = "cql",            # offline_q | cql | aptq
    epochs: int = 20,
    lr: float = 0.08,
    gamma: float = 0.95,
    cql_alpha: float = 0.20,
    aptq_initial_alpha: float = 0.20,
    aptq_alpha_lr: float = 0.02,
    aptq_target_source: str = "return_to_go_quantile",   # reward_quantile / dataset_q_quantile / fixed_value
    aptq_target_transform: str = "scale",
    aptq_target_scale: float = 0.10,
    aptq_target_offset: float = 0.0,
    aptq_target_quantile: float = 0.80,
    aptq_fixed_target: float = None,
    seed: int = 2026,
) -> Tuple[np.ndarray, List[dict]]:
    """Returns (Q-table, per-epoch history).

    Target sources for the adaptive coefficient (Eq. 6):
      * return_to_go_quantile / reward_quantile: static target from a data
        distribution (the proposed scaled-RTG variant uses the former).
      * dataset_q_quantile: MOVING quantile of the current learned dataset-Q
        distribution, recomputed each epoch. With quantile > 0.5 the update is
        systematically negative and drives alpha toward zero; this is a property
        of the moving target, not a faithful reproduction of a published
        fixed-target procedure.
      * fixed_value: a FIXED reference target tau passed in aptq_fixed_target
        (e.g. a quantile of a reference Q-function's dataset-Q distribution,
        computed once), giving a stationary target that alpha can converge to.
    """
    rng = random.Random(seed)
    q = np.zeros((len(state_index), num_actions), dtype=np.float64)
    alpha = {"offline_q": 0.0, "cql": cql_alpha, "aptq": aptq_initial_alpha}[algorithm]

    static_target = None
    if algorithm == "aptq" and aptq_target_source in ("reward_quantile", "return_to_go_quantile"):
        static_target = _static_target_q(
            train_rows, aptq_target_transform, aptq_target_scale, aptq_target_offset,
            aptq_target_quantile, aptq_target_source,
        )
    elif algorithm == "aptq" and aptq_target_source == "fixed_value":
        static_target = float(aptq_fixed_target)

    # precompute integer/float arrays once (major speed-up vs per-epoch dict lookups)
    arr = to_arrays(train_rows, state_index)
    S, NS, A, R, D = arr["s"], arr["ns"], arr["a"], arr["r"], arr["done"]
    n = len(S)
    order = np.arange(n)
    conservative = algorithm in ("cql", "aptq")
    history: List[dict] = []
    for epoch in range(1, epochs + 1):
        rng2 = np.random.default_rng(seed + epoch)
        rng2.shuffle(order)
        abs_sum = 0.0
        for i in order:
            s = S[i]; a = A[i]
            target = R[i] if D[i] else R[i] + gamma * q[NS[i]].max()
            td = target - q[s, a]
            q[s, a] += lr * td
            abs_sum += abs(td)
            if conservative and alpha > 0:
                row_q = q[s]
                m = row_q.max()
                e = np.exp(row_q - m)
                probs = e / e.sum()
                q[s] -= lr * alpha * probs
                q[s, a] += lr * alpha

        data_q = q[S, A]
        mean_dataset_q = float(data_q.mean()) if n else 0.0

        if algorithm == "aptq":
            if aptq_target_source == "dataset_q_quantile":   # moving-quantile baseline
                target_q = float(np.quantile(data_q, aptq_target_quantile)) if len(data_q) else 0.0
            else:
                target_q = static_target
            alpha = max(0.0, alpha + aptq_alpha_lr * (mean_dataset_q - target_q))
        else:
            target_q = ""

        history.append({
            "epoch": epoch, "algorithm": algorithm, "alpha": alpha,
            "target_q": target_q, "mean_dataset_q": mean_dataset_q,
            "mean_abs_td_error": (abs_sum / n) if n else 0.0,
        })
    return q, history


# --------------------------------------------------------------------------
# Deterministic-policy evaluation (unified for every method)
# --------------------------------------------------------------------------
def greedy_policy(q: np.ndarray) -> np.ndarray:
    return np.argmax(q, axis=1).astype(np.int64)


def bellman_mse(q: np.ndarray, rows: List[dict], state_index, gamma: float) -> float:
    errs = []
    for row in rows:
        s = state_index[row["state"]]; ns = state_index[row["next_state"]]; a = row["action"]
        tgt = row["reward"] if row["done"] else row["reward"] + gamma * float(np.max(q[ns]))
        errs.append((tgt - q[s, a]) ** 2)
    return float(np.mean(errs)) if errs else 0.0


def policy_metrics(
    policy: np.ndarray, rows: List[dict], state_index, counts: np.ndarray,
    behaviour: np.ndarray,
) -> dict:
    """Behaviour-agreement + count-based support diagnostics ."""
    matches, bin_support = [], []
    sel_counts, beh_probs = [], []
    tv_dist = []
    for row in rows:
        s = state_index[row["state"]]
        a_log = row["action"]
        a_pi = int(policy[s])
        matches.append(float(a_pi == a_log))
        bin_support.append(float(counts[s, a_pi] > 0))
        sel_counts.append(int(counts[s, a_pi]))
        beh_probs.append(float(behaviour[s, a_pi]))
        # TV distance between deterministic policy and behaviour policy at s
        onehot = np.zeros(behaviour.shape[1]); onehot[a_pi] = 1.0
        tv_dist.append(0.5 * float(np.sum(np.abs(onehot - behaviour[s]))))
    sc = np.array(sel_counts, dtype=np.float64)
    return {
        "transitions": len(rows),
        "action_match": float(np.mean(matches)) if matches else 0.0,
        "binary_in_support": float(np.mean(bin_support)) if bin_support else 0.0,
        "sel_count_median": float(np.median(sc)) if len(sc) else 0.0,
        "sel_count_min": float(np.min(sc)) if len(sc) else 0.0,
        "sel_count_q10": float(np.quantile(sc, 0.10)) if len(sc) else 0.0,
        "frac_sel_lt2": float(np.mean(sc < 2)) if len(sc) else 0.0,
        "frac_sel_lt5": float(np.mean(sc < 5)) if len(sc) else 0.0,
        "frac_sel_lt10": float(np.mean(sc < 10)) if len(sc) else 0.0,
        "frac_sel_lt20": float(np.mean(sc < 20)) if len(sc) else 0.0,
        "behaviour_prob_mean": float(np.mean(beh_probs)) if beh_probs else 0.0,
        "tv_distance_mean": float(np.mean(tv_dist)) if tv_dist else 0.0,
    }


# --------------------------------------------------------------------------
# Empirical (maximum-likelihood) MDP + model-based OPE (complementary to FQE, M2)
# --------------------------------------------------------------------------
def build_empirical_mdp(train_rows, state_index, num_actions):
    n = len(state_index)
    trans = defaultdict(lambda: defaultdict(int))   # (s,a) -> {s': count}
    rew_sum = np.zeros((n, num_actions)); sa_count = np.zeros((n, num_actions))
    term = np.zeros((n, num_actions))
    for row in train_rows:
        s = state_index[row["state"]]; ns = state_index[row["next_state"]]; a = row["action"]
        # Successor counts are accumulated for NON-terminal transitions only, so
        # that sum_{s'} trans[(s,a)][s'] / sa_count[s,a] = P(continue, s' | s,a).
        # Terminal transitions contribute reward and termination mass but no
        # bootstrap successor (v1.1 correction; v1.0 pooled all successors and
        # additionally multiplied by (1 - d), double-counting termination).
        if not row["done"]:
            trans[(s, a)][ns] += 1
        rew_sum[s, a] += row["reward"]; sa_count[s, a] += 1
        term[s, a] += row["done"]
    return trans, rew_sum, sa_count, term


def evaluate_policy_empirical_mdp(policy, trans, rew_sum, sa_count, term, n_states, gamma, iters=400, tol=1e-6):
    """Iterative policy evaluation of a deterministic policy on the empirical MDP."""
    V = np.zeros(n_states)
    reward = np.zeros(n_states); done_p = np.zeros(n_states)
    succ = [None] * n_states
    for s in range(n_states):
        a = int(policy[s])
        c = sa_count[s, a]
        if c == 0:
            succ[s] = None
            continue
        reward[s] = rew_sum[s, a] / c
        done_p[s] = term[s, a] / c
        nxt = trans[(s, a)]
        # nonterminal successor probabilities; they sum to 1 - done_p[s]
        succ[s] = [(ns, cnt / c) for ns, cnt in nxt.items()]
    for _ in range(iters):
        delta = 0.0
        newV = V.copy()
        for s in range(n_states):
            if succ[s] is None:
                newV[s] = reward[s]
                continue
            boot = 0.0
            for ns, p in succ[s]:
                boot += p * V[ns]
            newV[s] = reward[s] + gamma * boot
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < tol:
            break
    return V


# --------------------------------------------------------------------------
# FQE (matches tabular_fqe_ope.py) with configurable lr/epochs for sensitivity (M2)
# --------------------------------------------------------------------------
def fqe_fit(policy, train_rows, state_index, gamma, lr=0.05, epochs=60, seed=2026, num_actions=None):
    """Fit the FQE evaluation Q-function for a fixed deterministic policy."""
    if num_actions is None:
        max_logged = max((r["action"] for r in train_rows), default=0)
        max_pol = int(np.max(policy)) if len(policy) else 0
        num_actions = max(max_logged, max_pol) + 1
    q_eval = np.zeros((len(state_index), num_actions), dtype=np.float64)
    arr = to_arrays(train_rows, state_index)
    S, NS, A, R, D = arr["s"], arr["ns"], arr["a"], arr["r"], arr["done"]
    pol = np.asarray(policy)
    n = len(S); order = np.arange(n)
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        rng.shuffle(order)
        for i in order:
            tgt = R[i]
            if not D[i]:
                tgt += gamma * q_eval[NS[i], pol[NS[i]]]
            q_eval[S[i], A[i]] += lr * (tgt - q_eval[S[i], A[i]])
    return q_eval


def fqe_state_values(q_eval, eval_rows, policy, state_index):
    """Per-transition policy value Q_phi(s, pi(s)); used for mean and bootstrap."""
    pol = np.asarray(policy)
    return np.array([q_eval[state_index[r["state"]], pol[state_index[r["state"]]]] for r in eval_rows],
                    dtype=np.float64)


def fqe_value(policy, train_rows, eval_rows, state_index, gamma, lr=0.05, epochs=60, seed=2026,
              num_actions=None):
    q_eval = fqe_fit(policy, train_rows, state_index, gamma, lr, epochs, seed, num_actions)
    vals = fqe_state_values(q_eval, eval_rows, policy, state_index)
    return float(vals.mean()) if len(vals) else 0.0


# --------------------------------------------------------------------------
# Learner-level bootstrap
# --------------------------------------------------------------------------
def bootstrap_values_by_user(values, user_ids, n_boot=1000, seed=2026, alpha=0.05):
    """Fast learner bootstrap of the MEAN of precomputed per-transition values.

    `values` and `user_ids` are aligned arrays/lists over eval transitions.
    Resamples learners with replacement and averages the pooled transitions.
    """
    values = np.asarray(values, dtype=np.float64)
    groups: Dict[str, list] = defaultdict(list)
    for v, u in zip(values, user_ids):
        groups[u].append(v)
    users = list(groups.keys())
    user_arrays = [np.array(groups[u]) for u in users]
    rng = np.random.default_rng(seed)
    m = len(users)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        tot = 0.0; cnt = 0
        for j in pick:
            arr = user_arrays[j]
            tot += arr.sum(); cnt += arr.size
        stats[b] = tot / cnt if cnt else 0.0
    return float(stats.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def bootstrap_paired_by_user(values_a, values_b, user_ids, n_boot=1000, seed=2026, alpha=0.05):
    """Learner bootstrap of the mean paired difference (a - b) over transitions."""
    a = np.asarray(values_a, dtype=np.float64); b = np.asarray(values_b, dtype=np.float64)
    diff = a - b
    groups: Dict[str, list] = defaultdict(list)
    for d, u in zip(diff, user_ids):
        groups[u].append(d)
    users = list(groups.keys())
    user_arrays = [np.array(groups[u]) for u in users]
    rng = np.random.default_rng(seed)
    m = len(users)
    stats = np.empty(n_boot)
    for bi in range(n_boot):
        pick = rng.integers(0, m, size=m)
        tot = 0.0; cnt = 0
        for j in pick:
            arr = user_arrays[j]
            tot += arr.sum(); cnt += arr.size
        stats[bi] = tot / cnt if cnt else 0.0
    return float(stats.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def learner_bootstrap_ci(rows, metric_fn, n_boot=1000, seed=2026, alpha=0.05):
    """metric_fn(list_of_rows) -> float. Resamples learners with replacement."""
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    users = list(by_user.keys())
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample_rows = []
        for _ in range(len(users)):
            u = users[rng.randrange(len(users))]
            sample_rows.extend(by_user[u])
        stats.append(metric_fn(sample_rows))
    lo = float(np.quantile(stats, alpha / 2)); hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(np.mean(stats)), lo, hi


# --------------------------------------------------------------------------
# CSV writer
# --------------------------------------------------------------------------
def write_rows(path: Path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
