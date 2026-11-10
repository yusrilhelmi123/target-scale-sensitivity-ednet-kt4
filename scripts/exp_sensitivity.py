# -*- coding: utf-8 -*-
"""Learner-split and reward-rescaling sensitivity experiments.
Portable: paths are passed on the command line; `common` is imported from this
script's own directory. Run from anywhere.

Examples
--------
python exp_sensitivity.py ms --transitions-csv DATA.csv --out-dir out/ --split 1234 --seed 2026 --policy aptq
python exp_sensitivity.py ms --transitions-csv DATA.csv --out-dir out/ --split 1234 --seed 2026 --policy cql_matched --alpha 0.0457
python exp_sensitivity.py rs --transitions-csv DATA.csv --out-dir out/ --mult 0.5 --scale 0.10
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # relative import of common
import common as C

NA = 7; G = 0.95

def load(csv_path, mult=1.0):
    rows = C.read_transitions(csv_path)
    if mult != 1.0:
        for r in rows:
            r["reward"] = r["reward"] * mult
            r["return_to_go"] = r["return_to_go"] * mult
    return rows, C.build_state_index(rows)

def init_rows(rows):
    f = {}
    for r in rows:
        u = r["user_id"]
        if u not in f or r["timestep"] < f[u]["timestep"]:
            f[u] = r
    return list(f.values())

def amatch(q, rows, sidx):
    pol = C.greedy_policy(q)
    return float(np.mean([pol[sidx[r["state"]]] == r["action"] for r in rows]))

def train(rows, sidx, alg, alpha=0.20, scale=0.10, q=0.90, seed=2026):
    return C.train_q(rows, sidx, NA, algorithm=alg, epochs=20, lr=0.08, gamma=G, cql_alpha=alpha,
                     aptq_initial_alpha=0.20, aptq_alpha_lr=0.02, aptq_target_source="return_to_go_quantile",
                     aptq_target_transform="scale", aptq_target_scale=scale, aptq_target_quantile=q, seed=seed)

def appendl(out_dir, fn, rec):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fn), "a") as f:
        f.write(json.dumps(rec) + "\n")

def job_ms(csv, out, split_seed, seed, policy, matched_alpha=None):
    t0 = time.time(); rows, sidx = load(csv)
    tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, split_seed); te_init = init_rows(te)
    rec = {"split_seed": split_seed, "seed": seed, "policy": policy, "n_train": len(tr), "n_test": len(te)}
    if policy == "aptq":
        q, h = train(tr, sidx, "aptq", seed=seed); rec["final_alpha"] = h[-1]["alpha"]
        rec["val_bellman"] = C.bellman_mse(q, va, sidx, G)
        rec["fqe_init"] = C.fqe_value(C.greedy_policy(q), tr, te_init, sidx, G, seed=seed)
    elif policy == "cql_matched":
        q, _ = train(tr, sidx, "cql", alpha=float(matched_alpha), seed=seed); rec["alpha"] = float(matched_alpha)
    elif policy == "cql_020":
        q, _ = train(tr, sidx, "cql", alpha=0.20, seed=seed)
    elif policy == "offline_q":
        q, _ = train(tr, sidx, "offline_q", seed=seed)
    else:
        raise SystemExit("unknown policy " + policy)
    rec["bellman"] = C.bellman_mse(q, te, sidx, G); rec["match"] = amatch(q, te, sidx)
    rec["fqe"] = C.fqe_value(C.greedy_policy(q), tr, te, sidx, G, seed=seed)
    rec["rt"] = round(time.time() - t0, 1); appendl(out, "multisplit.jsonl", rec)
    print("OK ms", split_seed, seed, policy, "a=", rec.get("final_alpha", rec.get("alpha", "")),
          "bell=%.4f match=%.4f fqe=%.4f" % (rec["bellman"], rec["match"], rec["fqe"]), "rt", rec["rt"])
    return rec

def job_rs(csv, out, mult, scale, seed=2026, quant=0.90, split_seed=20260728):
    t0 = time.time(); rows, sidx = load(csv, mult)
    tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, split_seed)
    q, h = train(tr, sidx, "aptq", scale=scale, q=quant, seed=seed)
    rec = {"mult": mult, "scale": scale, "quant": quant, "seed": seed, "final_alpha": h[-1]["alpha"],
           "val_bellman": C.bellman_mse(q, va, sidx, G), "val_match": amatch(q, va, sidx), "rt": round(time.time() - t0, 1)}
    appendl(out, "reward_select.jsonl", rec)
    print("OK rs m=%.1f scale=%.3f q=%.2f fa=%.4f valbell=%.4f rt=%.1f" % (
        mult, scale, quant, rec["final_alpha"], rec["val_bellman"], rec["rt"]))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Learner-split and reward-rescaling sensitivity driver")
    p.add_argument("cmd", choices=["ms", "rs"])
    p.add_argument("--transitions-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", type=int, default=20260728)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--policy")
    p.add_argument("--alpha", type=float)
    p.add_argument("--mult", type=float)
    p.add_argument("--scale", type=float)
    p.add_argument("--quant", type=float, default=0.90)
    a = p.parse_args()
    if a.cmd == "ms":
        job_ms(a.transitions_csv, a.out_dir, a.split, a.seed, a.policy, a.alpha)
    elif a.cmd == "rs":
        job_rs(a.transitions_csv, a.out_dir, a.mult, a.scale, a.seed, a.quant, a.split)
