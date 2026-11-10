#!/usr/bin/env python
"""Leak-free 200k EdNet-KT4 experiments with matched CQL controls.

The runner has three resumable stages:

1. ``select``: create one fixed learner-disjoint 70/15/15 split and select
   hyperparameters using TRAIN/VALIDATION only. TEST is not evaluated.
2. ``evaluate``: train one stochastic seed using the frozen selection and
   report all metrics on the untouched TEST learners.
3. ``aggregate``: combine completed seed files and compute seed-level
   mean/std/95% Student-t intervals.

This script deliberately does not call either dataset-Q construction
"original APTQ". The moving-quantile and fixed-reference-target variants are
reported as two explicit interpretations because the earlier moving-target
implementation is not a faithful reproduction of a published fixed-target
procedure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
from exp_baselines import spibb_policy  # noqa: E402


CORE_FQE_IDS = {
    "offline_q",
    "cql_020",
    "cql_val_mse",
    "cql_action_match",
    "cql_final_alpha_matched",
    "aptq_norm_rtg",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(values):
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def initial_state_rows(rows: List[dict]) -> List[dict]:
    first: Dict[str, dict] = {}
    for row in rows:
        user = row["user_id"]
        if user not in first or row["timestep"] < first[user]["timestep"]:
            first[user] = row
    return list(first.values())


def split_fixed(rows: List[dict], split_seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    return C.split_by_user_three_way(
        rows, val_fraction=0.15, test_fraction=0.15, seed=split_seed
    )


def split_manifest(
    rows: List[dict],
    train: List[dict],
    val: List[dict],
    test: List[dict],
    dataset: Path,
    split_seed: int,
) -> dict:
    def describe(part: List[dict]) -> dict:
        users = sorted({row["user_id"] for row in part})
        return {
            "users": users,
            "user_count": len(users),
            "transition_count": len(part),
            "user_list_sha256": sha256_lines(users),
        }

    all_users = {row["user_id"] for row in rows}
    train_info, val_info, test_info = describe(train), describe(val), describe(test)
    assert not (set(train_info["users"]) & set(val_info["users"]))
    assert not (set(train_info["users"]) & set(test_info["users"]))
    assert not (set(val_info["users"]) & set(test_info["users"]))
    assert (
        set(train_info["users"]) | set(val_info["users"]) | set(test_info["users"])
    ) == all_users
    return {
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "users": len(all_users),
        "split_seed": split_seed,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "train": train_info,
        "validation": val_info,
        "test": test_info,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selection_stage(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = C.read_transitions(args.transitions_csv)
    train, val, test = split_fixed(rows, args.split_seed)
    sidx = C.build_state_index(rows)
    counts = C.state_action_counts(train, sidx, args.num_actions)
    behaviour = C.behaviour_policy_from_counts(counts)

    manifest = split_manifest(
        rows, train, val, test, args.transitions_csv, args.split_seed
    )
    write_json(out_dir / "split_manifest.json", manifest)

    grid_rows: List[dict] = []
    cql_results: List[dict] = []
    aptq_results: List[dict] = []

    print(
        f"[select] fixed split seed={args.split_seed}: "
        f"train={len(train)}, val={len(val)}, test={len(test)}",
        flush=True,
    )

    for alpha in args.cql_alphas:
        q, _ = C.train_q(
            train,
            sidx,
            args.num_actions,
            algorithm="cql",
            cql_alpha=alpha,
            epochs=args.epochs,
            lr=args.learning_rate,
            gamma=args.gamma,
            seed=args.selection_train_seed,
        )
        pi = C.greedy_policy(q)
        pm = C.policy_metrics(pi, val, sidx, counts, behaviour)
        record = {
            "family": "cql_fixed",
            "cql_alpha": alpha,
            "scale": "",
            "quantile": "",
            "eta": "",
            "initial_alpha": "",
            "selection_train_seed": args.selection_train_seed,
            "bellman_mse_val": C.bellman_mse(q, val, sidx, args.gamma),
            "action_match_val": pm["action_match"],
            "binary_in_support_val": pm["binary_in_support"],
            "tv_distance_val": pm["tv_distance_mean"],
            "final_alpha": alpha,
        }
        cql_results.append(record)
        grid_rows.append(record)
        print(
            f"[select] CQL alpha={alpha:.4f} "
            f"MSE={record['bellman_mse_val']:.6f} "
            f"match={record['action_match_val']:.6f}",
            flush=True,
        )

    for scale in args.aptq_scales:
        for quantile in args.aptq_quantiles:
            q, history = C.train_q(
                train,
                sidx,
                args.num_actions,
                algorithm="aptq",
                epochs=args.epochs,
                lr=args.learning_rate,
                gamma=args.gamma,
                aptq_initial_alpha=args.aptq_initial_alpha,
                aptq_alpha_lr=args.aptq_eta,
                aptq_target_source="return_to_go_quantile",
                aptq_target_transform="scale",
                aptq_target_scale=scale,
                aptq_target_quantile=quantile,
                seed=args.selection_train_seed,
            )
            pi = C.greedy_policy(q)
            pm = C.policy_metrics(pi, val, sidx, counts, behaviour)
            record = {
                "family": "aptq_norm_rtg",
                "cql_alpha": "",
                "scale": scale,
                "quantile": quantile,
                "eta": args.aptq_eta,
                "initial_alpha": args.aptq_initial_alpha,
                "selection_train_seed": args.selection_train_seed,
                "bellman_mse_val": C.bellman_mse(q, val, sidx, args.gamma),
                "action_match_val": pm["action_match"],
                "binary_in_support_val": pm["binary_in_support"],
                "tv_distance_val": pm["tv_distance_mean"],
                "final_alpha": history[-1]["alpha"],
            }
            aptq_results.append(record)
            grid_rows.append(record)
            print(
                f"[select] APTQ scale={scale:.3f} q={quantile:.2f} "
                f"MSE={record['bellman_mse_val']:.6f} "
                f"match={record['action_match_val']:.6f} "
                f"final_alpha={record['final_alpha']:.6f}",
                flush=True,
            )

    # SPIBB n_wedge selection is based only on validation empirical-MDP value.
    trans, rew_sum, sa_count, term = C.build_empirical_mdp(
        train, sidx, args.num_actions
    )
    spibb_results: List[dict] = []
    for n_wedge in args.spibb_n_wedges:
        pi, q = spibb_policy(
            counts,
            trans,
            rew_sum,
            sa_count,
            term,
            len(sidx),
            args.num_actions,
            args.gamma,
            n_wedge=n_wedge,
        )
        pm = C.policy_metrics(pi, val, sidx, counts, behaviour)
        values = C.evaluate_policy_empirical_mdp(
            pi,
            trans,
            rew_sum,
            sa_count,
            term,
            len(sidx),
            args.gamma,
        )
        emp_val = float(np.mean([values[sidx[r["state"]]] for r in val]))
        record = {
            "family": "spibb",
            "n_wedge": n_wedge,
            "selection_train_seed": args.selection_train_seed,
            "bellman_mse_val": C.bellman_mse(q, val, sidx, args.gamma),
            "action_match_val": pm["action_match"],
            "binary_in_support_val": pm["binary_in_support"],
            "tv_distance_val": pm["tv_distance_mean"],
            "empirical_mdp_value_val": emp_val,
        }
        spibb_results.append(record)
        grid_rows.append(record)

    best_aptq = min(
        aptq_results,
        key=lambda r: (r["bellman_mse_val"], -r["action_match_val"]),
    )
    best_cql = min(
        cql_results,
        key=lambda r: (r["bellman_mse_val"], -r["action_match_val"]),
    )
    match_cql = min(
        cql_results,
        key=lambda r: (
            abs(r["action_match_val"] - best_aptq["action_match_val"]),
            r["bellman_mse_val"],
        ),
    )
    closest_final_cql = min(
        cql_results,
        key=lambda r: (
            abs(r["cql_alpha"] - best_aptq["final_alpha"]),
            r["bellman_mse_val"],
        ),
    )
    best_spibb = max(
        spibb_results,
        key=lambda r: (r["empirical_mdp_value_val"], -r["bellman_mse_val"]),
    )

    selection = {
        "schema_version": 1,
        "dataset_sha256": manifest["dataset_sha256"],
        "split_seed": args.split_seed,
        "selection_train_seed": args.selection_train_seed,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "num_actions": args.num_actions,
        },
        "aptq_norm_rtg": {
            "scale": best_aptq["scale"],
            "quantile": best_aptq["quantile"],
            "eta": best_aptq["eta"],
            "initial_alpha": best_aptq["initial_alpha"],
            "validation_bellman_mse": best_aptq["bellman_mse_val"],
            "validation_action_match": best_aptq["action_match_val"],
            "selection_final_alpha": best_aptq["final_alpha"],
        },
        "cql_controls": {
            "reference_alpha": 0.20,
            "validation_mse_selected_alpha": best_cql["cql_alpha"],
            "action_match_selected_alpha": match_cql["cql_alpha"],
            "grid_alpha_closest_to_aptq_final": closest_final_cql["cql_alpha"],
        },
        "spibb": {
            "n_wedge": best_spibb["n_wedge"],
            "validation_empirical_mdp_value": best_spibb[
                "empirical_mdp_value_val"
            ],
        },
        "selection_rule": {
            "aptq": "minimum validation Bellman MSE; action match is tie-breaker",
            "cql_val_mse": "minimum validation Bellman MSE",
            "cql_action_match": "closest validation action match to selected APTQ",
            "spibb": "maximum validation empirical-MDP value",
        },
    }
    C.write_rows(out_dir / "selection_grid.csv", grid_rows)
    write_json(out_dir / "selection.json", selection)
    print(
        "[select] frozen selection written to "
        f"{out_dir / 'selection.json'}",
        flush=True,
    )


def policy_record(
    policy_id: str,
    policy: str,
    pi: np.ndarray,
    q: np.ndarray | None,
    *,
    cql_alpha: float | str = "",
    final_alpha: float | str = "",
) -> dict:
    return {
        "policy_id": policy_id,
        "policy": policy,
        "pi": pi,
        "q": q,
        "cql_alpha": cql_alpha,
        "final_alpha": final_alpha,
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection_json)
    rows = C.read_transitions(args.transitions_csv)
    dataset_hash = sha256_file(args.transitions_csv)
    if dataset_hash != selection["dataset_sha256"]:
        raise RuntimeError(
            "Dataset SHA256 differs from selection.json; refusing to mix runs."
        )

    split_seed = int(selection["split_seed"])
    train, val, test = split_fixed(rows, split_seed)
    sidx = C.build_state_index(rows)
    num_actions = int(selection["training"]["num_actions"])
    gamma = float(selection["training"]["gamma"])
    epochs = int(selection["training"]["epochs"])
    lr = float(selection["training"]["learning_rate"])
    counts = C.state_action_counts(train, sidx, num_actions)
    behaviour = C.behaviour_policy_from_counts(counts)
    test_init = initial_state_rows(test)
    test_users = [r["user_id"] for r in test]

    apt = selection["aptq_norm_rtg"]
    controls = selection["cql_controls"]
    seed = args.train_seed
    print(
        f"[evaluate] seed={seed}, fixed split={split_seed}, "
        f"train={len(train)}, val={len(val)}, test={len(test)}",
        flush=True,
    )

    policies: List[dict] = []
    histories: List[dict] = []
    cql_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    q_offline, h_offline = C.train_q(
        train,
        sidx,
        num_actions,
        algorithm="offline_q",
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        seed=seed,
    )
    pi_offline = C.greedy_policy(q_offline)
    policies.append(
        policy_record("offline_q", "Offline Q", pi_offline, q_offline)
    )

    q_aptq, h_aptq = C.train_q(
        train,
        sidx,
        num_actions,
        algorithm="aptq",
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        aptq_initial_alpha=float(apt["initial_alpha"]),
        aptq_alpha_lr=float(apt["eta"]),
        aptq_target_source="return_to_go_quantile",
        aptq_target_transform="scale",
        aptq_target_scale=float(apt["scale"]),
        aptq_target_quantile=float(apt["quantile"]),
        seed=seed,
    )
    aptq_final_alpha = float(h_aptq[-1]["alpha"])
    policies.append(
        policy_record(
            "aptq_norm_rtg",
            "APTQ-inspired scaled RTG",
            C.greedy_policy(q_aptq),
            q_aptq,
            final_alpha=aptq_final_alpha,
        )
    )

    def get_cql(alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        key = round(float(alpha), 12)
        if key not in cql_cache:
            q, _ = C.train_q(
                train,
                sidx,
                num_actions,
                algorithm="cql",
                cql_alpha=float(alpha),
                epochs=epochs,
                lr=lr,
                gamma=gamma,
                seed=seed,
            )
            cql_cache[key] = (C.greedy_policy(q), q)
        return cql_cache[key]

    control_specs = [
        ("cql_020", "CQL reference alpha=0.20", 0.20),
        (
            "cql_val_mse",
            "CQL validation-MSE-selected alpha",
            float(controls["validation_mse_selected_alpha"]),
        ),
        (
            "cql_action_match",
            "CQL validation-action-match-matched alpha",
            float(controls["action_match_selected_alpha"]),
        ),
        (
            "cql_final_alpha_matched",
            "CQL final-alpha-matched",
            aptq_final_alpha,
        ),
    ]
    for policy_id, name, alpha in control_specs:
        pi, q = get_cql(alpha)
        policies.append(
            policy_record(
                policy_id,
                name,
                pi,
                q,
                cql_alpha=alpha,
                final_alpha=alpha,
            )
        )

    q_reward, h_reward = C.train_q(
        train,
        sidx,
        num_actions,
        algorithm="aptq",
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        aptq_initial_alpha=0.20,
        aptq_alpha_lr=0.02,
        aptq_target_source="reward_quantile",
        aptq_target_transform="identity",
        aptq_target_quantile=0.60,
        seed=seed,
    )
    policies.append(
        policy_record(
            "aptq_reward_q",
            "APTQ-inspired reward quantile",
            C.greedy_policy(q_reward),
            q_reward,
            final_alpha=float(h_reward[-1]["alpha"]),
        )
    )

    q_moving, h_moving = C.train_q(
        train,
        sidx,
        num_actions,
        algorithm="aptq",
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        aptq_initial_alpha=0.20,
        aptq_alpha_lr=0.02,
        aptq_target_source="dataset_q_quantile",
        aptq_target_quantile=0.80,
        seed=seed,
    )
    policies.append(
        policy_record(
            "aptq_moving_dataset_q",
            "APTQ-style moving dataset-Q quantile",
            C.greedy_policy(q_moving),
            q_moving,
            final_alpha=float(h_moving[-1]["alpha"]),
        )
    )

    arr_train = C.to_arrays(train, sidx)
    ref_dataset_q = q_offline[arr_train["s"], arr_train["a"]]
    fixed_tau = float(np.quantile(ref_dataset_q, 0.80))
    q_fixed, h_fixed = C.train_q(
        train,
        sidx,
        num_actions,
        algorithm="aptq",
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        aptq_initial_alpha=0.20,
        aptq_alpha_lr=0.02,
        aptq_target_source="fixed_value",
        aptq_fixed_target=fixed_tau,
        seed=seed,
    )
    policies.append(
        policy_record(
            "aptq_fixed_reference_q",
            "APTQ-style fixed reference-Q target",
            C.greedy_policy(q_fixed),
            q_fixed,
            final_alpha=float(h_fixed[-1]["alpha"]),
        )
    )

    trans, rew_sum, sa_count, term = C.build_empirical_mdp(
        train, sidx, num_actions
    )
    n_wedge = int(selection["spibb"]["n_wedge"])
    pi_spibb, q_spibb = spibb_policy(
        counts,
        trans,
        rew_sum,
        sa_count,
        term,
        len(sidx),
        num_actions,
        gamma,
        n_wedge=n_wedge,
    )
    policies.append(
        policy_record(
            "spibb",
            f"SPIBB validation-selected n_wedge={n_wedge}",
            pi_spibb,
            q_spibb,
        )
    )

    majority_action = int(np.argmax(counts.sum(axis=0)))
    pi_bc = np.array(
        [
            int(np.argmax(counts[s])) if counts[s].sum() > 0 else majority_action
            for s in range(len(sidx))
        ],
        dtype=np.int64,
    )
    policies.append(policy_record("bc", "BC majority-action", pi_bc, None))
    policies.append(
        policy_record(
            "global_majority",
            "Global-majority",
            np.full(len(sidx), majority_action, dtype=np.int64),
            None,
        )
    )
    rng = np.random.default_rng(seed)
    policies.append(
        policy_record(
            "random",
            "Random",
            rng.integers(0, num_actions, size=len(sidx)).astype(np.int64),
            None,
        )
    )

    for policy_id, hist in [
        ("offline_q", h_offline),
        ("aptq_norm_rtg", h_aptq),
        ("aptq_reward_q", h_reward),
        ("aptq_moving_dataset_q", h_moving),
        ("aptq_fixed_reference_q", h_fixed),
    ]:
        for row in hist:
            histories.append({"policy_id": policy_id, "seed": seed, **row})

    metrics_rows: List[dict] = []
    paired_rows: List[dict] = []
    fqe_values: Dict[str, np.ndarray] = {}
    match_values: Dict[str, np.ndarray] = {}
    fqe_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for record in policies:
        policy_id = record["policy_id"]
        pi = record["pi"]
        q = record["q"]
        pm = C.policy_metrics(pi, test, sidx, counts, behaviour)
        empirical_values = C.evaluate_policy_empirical_mdp(
            pi,
            trans,
            rew_sum,
            sa_count,
            term,
            len(sidx),
            gamma,
        )
        emp_visit = float(
            np.mean([empirical_values[sidx[r["state"]]] for r in test])
        )
        emp_init = float(
            np.mean([empirical_values[sidx[r["state"]]] for r in test_init])
        )
        match_arr = np.array(
            [
                float(int(pi[sidx[row["state"]]]) == row["action"])
                for row in test
            ],
            dtype=np.float64,
        )
        match_values[policy_id] = match_arr
        match_boot = C.bootstrap_values_by_user(
            match_arr, test_users, n_boot=args.n_boot, seed=seed + 1000
        )

        fqe_visit = fqe_init = ""
        fqe_boot_mean = fqe_boot_lo = fqe_boot_hi = ""
        if policy_id in CORE_FQE_IDS:
            policy_hash = hashlib.sha256(pi.tobytes()).hexdigest()
            if policy_hash not in fqe_cache:
                q_eval = C.fqe_fit(
                    pi,
                    train,
                    sidx,
                    gamma,
                    lr=args.fqe_learning_rate,
                    epochs=args.fqe_epochs,
                    seed=seed,
                    num_actions=num_actions,
                )
                vals_test = C.fqe_state_values(q_eval, test, pi, sidx)
                vals_init = C.fqe_state_values(q_eval, test_init, pi, sidx)
                fqe_cache[policy_hash] = (q_eval, vals_test, vals_init)
            _, vals_test, vals_init = fqe_cache[policy_hash]
            fqe_values[policy_id] = vals_test
            fqe_visit = float(np.mean(vals_test))
            fqe_init = float(np.mean(vals_init))
            fqe_boot = C.bootstrap_values_by_user(
                vals_test, test_users, n_boot=args.n_boot, seed=seed + 2000
            )
            fqe_boot_mean, fqe_boot_lo, fqe_boot_hi = fqe_boot

        metrics_rows.append(
            {
                "seed": seed,
                "split_seed": split_seed,
                "policy_id": policy_id,
                "policy": record["policy"],
                "cql_alpha": record["cql_alpha"],
                "final_alpha": record["final_alpha"],
                "fixed_reference_tau": (
                    fixed_tau if policy_id == "aptq_fixed_reference_q" else ""
                ),
                "test_transitions": len(test),
                "test_users": len({r["user_id"] for r in test}),
                "bellman_mse": (
                    C.bellman_mse(q, test, sidx, gamma) if q is not None else ""
                ),
                "action_match": pm["action_match"],
                "binary_in_support": pm["binary_in_support"],
                "support_ge2": 1.0 - pm["frac_sel_lt2"],
                "support_ge5": 1.0 - pm["frac_sel_lt5"],
                "support_ge10": 1.0 - pm["frac_sel_lt10"],
                "support_ge20": 1.0 - pm["frac_sel_lt20"],
                "behaviour_prob_mean": pm["behaviour_prob_mean"],
                "tv_distance_mean": pm["tv_distance_mean"],
                "fqe_value_visitation": fqe_visit,
                "fqe_value_initial": fqe_init,
                "fqe_boot_mean": fqe_boot_mean,
                "fqe_boot_lo": fqe_boot_lo,
                "fqe_boot_hi": fqe_boot_hi,
                "match_boot_mean": match_boot[0],
                "match_boot_lo": match_boot[1],
                "match_boot_hi": match_boot[2],
                "empirical_mdp_value_visitation": emp_visit,
                "empirical_mdp_value_initial": emp_init,
            }
        )
        print(
            f"[evaluate] seed={seed} policy={policy_id} "
            f"MSE={metrics_rows[-1]['bellman_mse']} "
            f"match={pm['action_match']:.6f} FQE={fqe_visit}",
            flush=True,
        )

    aptq_id = "aptq_norm_rtg"
    for control_id in (
        "cql_020",
        "cql_val_mse",
        "cql_action_match",
        "cql_final_alpha_matched",
    ):
        match_delta = C.bootstrap_paired_by_user(
            match_values[aptq_id],
            match_values[control_id],
            test_users,
            n_boot=args.n_boot,
            seed=seed + 3000,
        )
        fqe_delta = C.bootstrap_paired_by_user(
            fqe_values[aptq_id],
            fqe_values[control_id],
            test_users,
            n_boot=args.n_boot,
            seed=seed + 4000,
        )
        paired_rows.append(
            {
                "seed": seed,
                "aptq_policy_id": aptq_id,
                "control_policy_id": control_id,
                "delta_match_mean": match_delta[0],
                "delta_match_lo": match_delta[1],
                "delta_match_hi": match_delta[2],
                "delta_fqe_mean": fqe_delta[0],
                "delta_fqe_lo": fqe_delta[1],
                "delta_fqe_hi": fqe_delta[2],
            }
        )

    C.write_rows(out_dir / f"seed_{seed}_metrics.csv", metrics_rows)
    C.write_rows(out_dir / f"seed_{seed}_paired.csv", paired_rows)
    C.write_rows(out_dir / f"seed_{seed}_histories.csv", histories)
    np.savez_compressed(
        out_dir / f"seed_{seed}_q_tables.npz",
        **{
            record["policy_id"]: record["q"]
            for record in policies
            if record["q"] is not None
        },
    )
    write_json(
        out_dir / f"seed_{seed}_run.json",
        {
            "seed": seed,
            "split_seed": split_seed,
            "dataset_sha256": dataset_hash,
            "selection_json": str(Path(args.selection_json).resolve()),
            "fqe": {
                "learning_rate": args.fqe_learning_rate,
                "epochs": args.fqe_epochs,
            },
            "bootstrap_replicates": args.n_boot,
            "status": "complete",
        },
    )
    print(f"[evaluate] seed={seed} complete", flush=True)


def complete_fqe_stage(args: argparse.Namespace) -> None:
    """Fill FQE fields for policies intentionally skipped in the main run.

    Q-learning is not repeated. Learned policies are reconstructed from the
    saved Q-table archive; calibration policies are deterministically rebuilt
    from the fixed split and seed.
    """
    out_dir = Path(args.out_dir)
    selection = read_json(args.selection_json)
    rows = C.read_transitions(args.transitions_csv)
    if sha256_file(args.transitions_csv) != selection["dataset_sha256"]:
        raise RuntimeError(
            "Dataset SHA256 differs from selection.json; refusing to mix runs."
        )

    seed = args.train_seed
    metrics_path = out_dir / f"seed_{seed}_metrics.csv"
    q_tables_path = out_dir / f"seed_{seed}_q_tables.npz"
    if not metrics_path.exists() or not q_tables_path.exists():
        raise RuntimeError(f"Seed {seed} main outputs are incomplete.")

    train, _, test = split_fixed(rows, int(selection["split_seed"]))
    test_init = initial_state_rows(test)
    test_users = [r["user_id"] for r in test]
    sidx = C.build_state_index(rows)
    num_actions = int(selection["training"]["num_actions"])
    gamma = float(selection["training"]["gamma"])
    counts = C.state_action_counts(train, sidx, num_actions)
    majority_action = int(np.argmax(counts.sum(axis=0)))

    metrics_rows = read_csv_rows(metrics_path)
    with np.load(q_tables_path) as archive:
        q_tables = {name: archive[name] for name in archive.files}

    policy_cache: Dict[str, np.ndarray] = {}
    for row in metrics_rows:
        policy_id = row["policy_id"]
        if policy_id in q_tables:
            policy_cache[policy_id] = C.greedy_policy(q_tables[policy_id])
        elif policy_id == "bc":
            policy_cache[policy_id] = np.array(
                [
                    int(np.argmax(counts[s]))
                    if counts[s].sum() > 0
                    else majority_action
                    for s in range(len(sidx))
                ],
                dtype=np.int64,
            )
        elif policy_id == "global_majority":
            policy_cache[policy_id] = np.full(
                len(sidx), majority_action, dtype=np.int64
            )
        elif policy_id == "random":
            rng = np.random.default_rng(seed)
            policy_cache[policy_id] = rng.integers(
                0, num_actions, size=len(sidx)
            ).astype(np.int64)
        else:
            raise RuntimeError(f"Cannot reconstruct policy {policy_id}")

    fqe_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for row in metrics_rows:
        if row.get("fqe_value_visitation", "") != "":
            continue
        policy_id = row["policy_id"]
        pi = policy_cache[policy_id]
        policy_hash = hashlib.sha256(pi.tobytes()).hexdigest()
        if policy_hash not in fqe_cache:
            q_eval = C.fqe_fit(
                pi,
                train,
                sidx,
                gamma,
                lr=args.fqe_learning_rate,
                epochs=args.fqe_epochs,
                seed=seed,
                num_actions=num_actions,
            )
            vals_test = C.fqe_state_values(q_eval, test, pi, sidx)
            vals_init = C.fqe_state_values(q_eval, test_init, pi, sidx)
            fqe_cache[policy_hash] = (vals_test, vals_init)
        vals_test, vals_init = fqe_cache[policy_hash]
        boot = C.bootstrap_values_by_user(
            vals_test, test_users, n_boot=args.n_boot, seed=seed + 5000
        )
        row["fqe_value_visitation"] = float(np.mean(vals_test))
        row["fqe_value_initial"] = float(np.mean(vals_init))
        row["fqe_boot_mean"] = boot[0]
        row["fqe_boot_lo"] = boot[1]
        row["fqe_boot_hi"] = boot[2]
        print(
            f"[complete-fqe] seed={seed} policy={policy_id} "
            f"FQE={row['fqe_value_visitation']:.6f}",
            flush=True,
        )

    C.write_rows(metrics_path, metrics_rows)
    write_json(
        out_dir / f"seed_{seed}_fqe_all.json",
        {
            "seed": seed,
            "status": "complete",
            "fqe_learning_rate": args.fqe_learning_rate,
            "fqe_epochs": args.fqe_epochs,
            "bootstrap_replicates": args.n_boot,
            "filled_policy_ids": sorted(
                row["policy_id"]
                for row in metrics_rows
                if row.get("fqe_value_visitation", "") != ""
            ),
        },
    )
    print(f"[complete-fqe] seed={seed} complete", flush=True)


def parse_number(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 15:
        return table[10]
    if df < 20:
        return table[15]
    if df < 30:
        return table[20]
    return 1.96


def summarize_group(rows: List[dict], keys: List[str]) -> dict:
    out: dict = {}
    n = len(rows)
    out["n_seeds"] = n
    for key in keys:
        vals = [parse_number(row.get(key, "")) for row in rows]
        values = np.array([v for v in vals if v is not None], dtype=np.float64)
        if len(values) == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        half = (
            t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
        out[f"{key}_ci95_lo"] = mean - half
        out[f"{key}_ci95_hi"] = mean + half
    return out


def aggregate_stage(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    metric_files = sorted(out_dir.glob("seed_*_metrics.csv"))
    paired_files = sorted(out_dir.glob("seed_*_paired.csv"))
    if not metric_files:
        raise RuntimeError(f"No seed metrics found in {out_dir}")

    metrics = [row for path in metric_files for row in read_csv_rows(path)]
    paired = [row for path in paired_files for row in read_csv_rows(path)]
    metric_lookup = {
        (row["seed"], row["policy_id"]): row
        for row in metrics
    }
    for row in paired:
        aptq = metric_lookup[(row["seed"], row["aptq_policy_id"])]
        control = metric_lookup[(row["seed"], row["control_policy_id"])]
        for metric in ("bellman_mse", "action_match", "fqe_value_visitation"):
            a = parse_number(aptq.get(metric, ""))
            c = parse_number(control.get(metric, ""))
            row[f"delta_{metric}_point"] = (
                (a - c) if a is not None and c is not None else ""
            )
    metrics_by_policy: Dict[str, List[dict]] = defaultdict(list)
    for row in metrics:
        metrics_by_policy[row["policy_id"]].append(row)

    metric_keys = [
        "cql_alpha",
        "final_alpha",
        "bellman_mse",
        "action_match",
        "binary_in_support",
        "support_ge2",
        "support_ge5",
        "support_ge10",
        "support_ge20",
        "behaviour_prob_mean",
        "tv_distance_mean",
        "fqe_value_visitation",
        "fqe_value_initial",
        "empirical_mdp_value_visitation",
        "empirical_mdp_value_initial",
    ]
    summary_rows = []
    for policy_id, rows in sorted(metrics_by_policy.items()):
        summary_rows.append(
            {
                "policy_id": policy_id,
                "policy": rows[0]["policy"],
                **summarize_group(rows, metric_keys),
            }
        )

    paired_by_control: Dict[str, List[dict]] = defaultdict(list)
    for row in paired:
        paired_by_control[row["control_policy_id"]].append(row)
    paired_keys = [
        "delta_bellman_mse_point",
        "delta_action_match_point",
        "delta_fqe_value_visitation_point",
        "delta_match_mean",
        "delta_fqe_mean",
    ]
    paired_summary = []
    for control_id, rows in sorted(paired_by_control.items()):
        paired_summary.append(
            {
                "aptq_policy_id": rows[0]["aptq_policy_id"],
                "control_policy_id": control_id,
                **summarize_group(rows, paired_keys),
            }
        )

    C.write_rows(out_dir / "all_seed_metrics.csv", metrics)
    C.write_rows(out_dir / "summary_metrics.csv", summary_rows)
    C.write_rows(out_dir / "all_seed_paired.csv", paired)
    C.write_rows(out_dir / "summary_paired.csv", paired_summary)
    write_json(
        out_dir / "aggregate_manifest.json",
        {
            "metric_files": [path.name for path in metric_files],
            "paired_files": [path.name for path in paired_files],
            "completed_seeds": sorted(
                {int(float(row["seed"])) for row in metrics}
            ),
            "status": "complete",
        },
    )
    print(
        f"[aggregate] {len(metric_files)} seeds -> "
        f"{out_dir / 'summary_metrics.csv'}",
        flush=True,
    )


def add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transitions-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="Validation-only hyperparameter selection")
    add_shared_data_args(select)
    select.add_argument("--split-seed", type=int, default=20260728)
    select.add_argument("--selection-train-seed", type=int, default=2026)
    select.add_argument("--num-actions", type=int, default=7)
    select.add_argument("--gamma", type=float, default=0.95)
    select.add_argument("--epochs", type=int, default=20)
    select.add_argument("--learning-rate", type=float, default=0.08)
    select.add_argument(
        "--cql-alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00],
    )
    select.add_argument(
        "--aptq-scales", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.50]
    )
    select.add_argument(
        "--aptq-quantiles", type=float, nargs="+", default=[0.60, 0.80, 0.90]
    )
    select.add_argument("--aptq-eta", type=float, default=0.02)
    select.add_argument("--aptq-initial-alpha", type=float, default=0.20)
    select.add_argument(
        "--spibb-n-wedges", type=int, nargs="+", default=[2, 5, 10, 20, 50]
    )
    select.set_defaults(func=selection_stage)

    evaluate = sub.add_parser("evaluate", help="Evaluate one train seed on TEST")
    add_shared_data_args(evaluate)
    evaluate.add_argument("--selection-json", required=True, type=Path)
    evaluate.add_argument("--train-seed", required=True, type=int)
    evaluate.add_argument("--fqe-learning-rate", type=float, default=0.05)
    evaluate.add_argument("--fqe-epochs", type=int, default=60)
    evaluate.add_argument("--n-boot", type=int, default=1000)
    evaluate.set_defaults(func=evaluate_stage)

    complete_fqe = sub.add_parser(
        "complete-fqe", help="Fill FQE for all remaining saved policies"
    )
    add_shared_data_args(complete_fqe)
    complete_fqe.add_argument("--selection-json", required=True, type=Path)
    complete_fqe.add_argument("--train-seed", required=True, type=int)
    complete_fqe.add_argument("--fqe-learning-rate", type=float, default=0.05)
    complete_fqe.add_argument("--fqe-epochs", type=int, default=60)
    complete_fqe.add_argument("--n-boot", type=int, default=1000)
    complete_fqe.set_defaults(func=complete_fqe_stage)

    aggregate = sub.add_parser("aggregate", help="Aggregate completed seed outputs")
    aggregate.add_argument("--out-dir", required=True, type=Path)
    aggregate.set_defaults(func=aggregate_stage)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
