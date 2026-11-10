"""v1.1 correction of the empirical-MDP evaluation and SPIBB planner.

Re-evaluates every archived policy of the clean-200k run with the corrected
(non-terminal-successor) empirical MDP, re-runs the validation-only SPIBB
n_wedge selection and the SPIBB policy, and writes corrected per-seed metric
files. Learned Q-tables are reused from the archived v1.0 run (they do not
depend on the empirical MDP). The v1.0 formula is also recomputed to confirm
that the archived numbers are reproduced before the correction is applied.
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import common as C, fastrl as F
from exp_baselines import spibb_policy
from clean_200k import initial_state_rows, split_fixed

# usage: python empirical_mdp_v11.py DATA.csv results_clean_200k_v10/ out_dir/ scripts/legacy/common_v10.py
DATA, OLD, OUT, OLDCOMMON = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("common_v10", OLDCOMMON); C0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(C0)
G, NA = 0.95, 7
rows = C.read_transitions(DATA); sidx = C.build_state_index(rows); n = len(sidx)
train, val, test = split_fixed(rows, 20260728)
counts = C.state_action_counts(train, sidx, NA); behaviour = C.behaviour_policy_from_counts(counts)
test_init = initial_state_rows(test); test_users = [r["user_id"] for r in test]
arr = C.to_arrays(train, sidx)
new_mdp = C.build_empirical_mdp(train, sidx, NA)
old_mdp = C0.build_empirical_mdp(train, sidx, NA)

def emp(pi, mdp, mod):
    V = mod.evaluate_policy_empirical_mdp(pi, *mdp, n, G)
    return (float(np.mean([V[sidx[r["state"]]] for r in test])),
            float(np.mean([V[sidx[r["state"]]] for r in test_init])),
            float(np.mean([V[sidx[r["state"]]] for r in val])))

# --- SPIBB validation selection with the corrected planner/evaluator
sel = []
for nw in [2, 5, 10, 20, 50]:
    pi, q = spibb_policy(counts, *new_mdp, n, NA, G, n_wedge=nw)
    V = C.evaluate_policy_empirical_mdp(pi, *new_mdp, n, G)
    sel.append({"n_wedge": nw, "empirical_mdp_value_val": float(np.mean([V[sidx[r["state"]]] for r in val])),
                "bellman_mse_val": C.bellman_mse(q, val, sidx, G),
                "action_match_val": C.policy_metrics(pi, val, sidx, counts, behaviour)["action_match"]})
    print("[spibb-select]", sel[-1], flush=True)
best = max(sel, key=lambda r: (r["empirical_mdp_value_val"], -r["bellman_mse_val"]))
pi_spibb, q_spibb = spibb_policy(counts, *new_mdp, n, NA, G, n_wedge=best["n_wedge"])
pd.DataFrame(sel).to_csv(OUT / "spibb_selection_v11.csv", index=False)

check = []
for seed in [2026, 2027, 2028, 2029, 2030]:
    m = pd.read_csv(OLD / f"seed_{seed}_metrics.csv")
    qt = np.load(OLD / f"seed_{seed}_q_tables.npz", allow_pickle=True)
    maj = int(np.argmax(counts.sum(axis=0)))
    pols = {k: C.greedy_policy(qt[k]) for k in qt.files if k != "spibb"}
    pols["bc"] = np.array([int(np.argmax(counts[s])) if counts[s].sum() > 0 else maj for s in range(n)], dtype=np.int64)
    pols["global_majority"] = np.full(n, maj, dtype=np.int64)
    pols["random"] = np.random.default_rng(seed).integers(0, NA, size=n).astype(np.int64)
    for i, r in m.iterrows():
        pid = r.policy_id
        if pid == "spibb":
            continue
        o_vis, o_init, _ = emp(pols[pid], old_mdp, C0)
        check.append({"seed": seed, "policy_id": pid, "archived_vis": r.empirical_mdp_value_visitation, "v10_recomputed_vis": o_vis,
                      "archived_init": r.empirical_mdp_value_initial, "v10_recomputed_init": o_init})
        n_vis, n_init, _ = emp(pols[pid], new_mdp, C)
        m.loc[i, "empirical_mdp_value_visitation"] = n_vis
        m.loc[i, "empirical_mdp_value_initial"] = n_init
    # SPIBB row: full recomputation with corrected planner
    i = m.index[m.policy_id == "spibb"][0]
    pm = C.policy_metrics(pi_spibb, test, sidx, counts, behaviour)
    qe = F.fqe_fit(pi_spibb, train, sidx, G, seed=seed, arrays=arr)
    vt = C.fqe_state_values(qe, test, pi_spibb, sidx); vi = C.fqe_state_values(qe, test_init, pi_spibb, sidx)
    fb = C.bootstrap_values_by_user(vt, test_users, n_boot=1000, seed=seed + 2000)
    ma = np.array([float(int(pi_spibb[sidx[r["state"]]]) == r["action"]) for r in test])
    mb = C.bootstrap_values_by_user(ma, test_users, n_boot=1000, seed=seed + 1000)
    n_vis, n_init, _ = emp(pi_spibb, new_mdp, C)
    upd = {"policy": f"SPIBB validation-selected n_wedge={best['n_wedge']}", "bellman_mse": C.bellman_mse(q_spibb, test, sidx, G),
           "action_match": pm["action_match"], "binary_in_support": pm["binary_in_support"],
           "support_ge2": 1 - pm["frac_sel_lt2"], "support_ge5": 1 - pm["frac_sel_lt5"], "support_ge10": 1 - pm["frac_sel_lt10"],
           "support_ge20": 1 - pm["frac_sel_lt20"], "behaviour_prob_mean": pm["behaviour_prob_mean"], "tv_distance_mean": pm["tv_distance_mean"],
           "fqe_value_visitation": float(vt.mean()), "fqe_value_initial": float(vi.mean()),
           "fqe_boot_mean": fb[0], "fqe_boot_lo": fb[1], "fqe_boot_hi": fb[2],
           "match_boot_mean": mb[0], "match_boot_lo": mb[1], "match_boot_hi": mb[2],
           "empirical_mdp_value_visitation": n_vis, "empirical_mdp_value_initial": n_init}
    for k, v in upd.items():
        m.loc[i, k] = v
    m.to_csv(OUT / f"seed_{seed}_metrics.csv", index=False)
    print("[seed]", seed, "done", flush=True)
ck = pd.DataFrame(check); ck.to_csv(OUT / "v10_reproduction_check.csv", index=False)
print("max |archived - v1.0 recomputed| vis:", (ck.archived_vis - ck.v10_recomputed_vis).abs().max(),
      "init:", (ck.archived_init - ck.v10_recomputed_init).abs().max())
json.dump({"spibb_n_wedge_v11": best["n_wedge"], "spibb_selection": sel}, open(OUT / "spibb_v11.json", "w"), indent=1)
