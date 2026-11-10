"""v1.1 additional controls (September 2026).

gamma : RTG-discount ablation. The RTG used to build the scaled target is
        recomputed with gamma_rtg in {0.99 (archived), 0.95 (= training/FQE
        discount)}; the target scale/quantile is re-selected on validation
        (grid includes an unscaled target, scale 1.0), then five training seeds
        are evaluated on the untouched test learners together with a
        final-alpha-matched CQL control and the validation-selected CQL(0.05).
reward: reward-scale controls. For reward multipliers {0.5, 1, 2} the target
        scale is re-selected on validation (as in the archived sweep); for the
        selected and the fixed (0.10) scale, five seeds are evaluated with a
        final-alpha-matched CQL control, fixed CQL(0.05) (unit-reward
        selection) and CQL with alpha re-selected on validation at that
        reward unit. Bellman MSE is reported divided by mult^2 and FQE divided
        by mult so that values are on the unit-reward scale.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import common as C, fastrl as F

DATA, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
NA, G, SPLIT = 7, 0.95, 20260728
SEEDS = [2026, 2027, 2028, 2029, 2030]
CQL_GRID = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

rows0 = C.read_transitions(DATA)
sidx = C.build_state_index(rows0)

def add_rtg(rows, g, key):
    by = {}
    for r in rows:
        by.setdefault(r["user_id"], []).append(r)
    for u, lst in by.items():
        lst.sort(key=lambda r: r["timestep"]); acc = 0.0
        for r in reversed(lst):
            acc = r["reward"] + g * acc; r[key] = acc

def prep(mult=1.0):
    rows = [dict(r) for r in rows0]
    for r in rows:
        r["reward"] *= mult; r["return_to_go"] *= mult
    add_rtg(rows, 0.95, "rtg095")
    tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, SPLIT)
    counts = C.state_action_counts(tr, sidx, NA)
    return tr, va, te, counts, C.behaviour_policy_from_counts(counts), C.to_arrays(tr, sidx)

def evalq(q, rows, counts, beh, tr, arr, seed, mult, fqe=True):
    pi = C.greedy_policy(q)
    pm = C.policy_metrics(pi, rows, sidx, counts, beh)
    out = {"action_match": pm["action_match"], "bellman_mse_unit": C.bellman_mse(q, rows, sidx, G) / mult**2,
           "support_ge20": 1 - pm["frac_sel_lt20"], "tv": pm["tv_distance_mean"]}
    if fqe:
        out["fqe_unit"] = F.fqe_value(pi, tr, rows, sidx, G, seed=seed, arrays=arr) / mult
    return out

def log(fn, rec):
    with open(OUT / fn, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:300], flush=True)

def aptq(tr, sidx_, arr, scale, quant, seed, key):
    return F.train_q(tr, sidx_, NA, algorithm="aptq", aptq_target_scale=scale, aptq_target_quantile=quant,
                     seed=seed, rtg_key=key, arrays=arr)

def run_gamma():
    tr, va, te, counts, beh, arr = prep()
    for key, g in [("return_to_go", 0.99), ("rtg095", 0.95)]:
        vals = np.array([r[key] for r in tr])
        grid = []
        for sc in [0.05, 0.10, 0.20, 0.50, 1.00]:
            for qu in [0.60, 0.80, 0.90]:
                q, h = aptq(tr, sidx, arr, sc, qu, 2026, key)
                rec = {"stage": "select", "gamma_rtg": g, "scale": sc, "quantile": qu,
                       "tau": float(np.quantile(vals * sc, qu)), "final_alpha": h[-1]["alpha"],
                       "val_bellman": C.bellman_mse(q, va, sidx, G),
                       "val_match": C.policy_metrics(C.greedy_policy(q), va, sidx, counts, beh)["action_match"]}
                grid.append(rec); log("gamma.jsonl", rec)
        best = min(grid, key=lambda r: (r["val_bellman"], -r["val_match"]))
        for seed in SEEDS:
            q, h = aptq(tr, sidx, arr, best["scale"], best["quantile"], seed, key)
            fa = h[-1]["alpha"]
            qm, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=fa, seed=seed, arrays=arr)
            qv, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=0.05, seed=seed, arrays=arr)
            log("gamma.jsonl", {"stage": "test", "gamma_rtg": g, "scale": best["scale"], "quantile": best["quantile"],
                                "seed": seed, "final_alpha": fa,
                                "aptq": evalq(q, te, counts, beh, tr, arr, seed, 1.0),
                                "cql_matched": evalq(qm, te, counts, beh, tr, arr, seed, 1.0),
                                "cql_005": evalq(qv, te, counts, beh, tr, arr, seed, 1.0)})

def run_reward():
    for mult in [0.5, 1.0, 2.0]:
        tr, va, te, counts, beh, arr = prep(mult)
        grid = []
        for sc in [0.05, 0.10, 0.20]:
            q, h = aptq(tr, sidx, arr, sc, 0.90, 2026, "return_to_go")
            rec = {"stage": "select_aptq", "mult": mult, "scale": sc, "final_alpha": h[-1]["alpha"],
                   "val_bellman": C.bellman_mse(q, va, sidx, G)}
            grid.append(rec); log("reward.jsonl", rec)
        best = min(grid, key=lambda r: r["val_bellman"])
        cgrid = []
        for a in CQL_GRID:
            q, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=a, seed=2026, arrays=arr)
            rec = {"stage": "select_cql", "mult": mult, "alpha": a, "val_bellman": C.bellman_mse(q, va, sidx, G)}
            cgrid.append(rec); log("reward.jsonl", rec)
        a_sel = min(cgrid, key=lambda r: r["val_bellman"])["alpha"]
        for sc_kind, sc in [("selected", best["scale"]), ("fixed", 0.10)]:
            if sc_kind == "fixed" and sc == best["scale"]:
                continue
            for seed in SEEDS:
                q, h = aptq(tr, sidx, arr, sc, 0.90, seed, "return_to_go"); fa = h[-1]["alpha"]
                qm, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=fa, seed=seed, arrays=arr)
                rec = {"stage": "test", "mult": mult, "scale_kind": sc_kind, "scale": sc, "seed": seed, "final_alpha": fa,
                       "aptq": evalq(q, te, counts, beh, tr, arr, seed, mult),
                       "cql_matched": evalq(qm, te, counts, beh, tr, arr, seed, mult)}
                log("reward.jsonl", rec)
        for seed in SEEDS:
            q5, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=0.05, seed=seed, arrays=arr)
            qs, _ = F.train_q(tr, sidx, NA, algorithm="cql", cql_alpha=a_sel, seed=seed, arrays=arr)
            log("reward.jsonl", {"stage": "test_cql", "mult": mult, "seed": seed, "alpha_reselected": a_sel,
                                 "cql_005": evalq(q5, te, counts, beh, tr, arr, seed, mult),
                                 "cql_reselected": evalq(qs, te, counts, beh, tr, arr, seed, mult)})

{"gamma": run_gamma, "reward": run_reward}[MODE]()
print("DONE", MODE)
