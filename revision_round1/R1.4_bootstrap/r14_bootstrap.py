import sys, csv, json, math
from pathlib import Path
import numpy as np, pandas as pd
ARC = Path(sys.argv[1]); DATA = sys.argv[2]; Q13 = Path(sys.argv[3]); OUT = Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
import common as C, fastrl as F, clean_200k as K

# ---- 1. what the current Table S2 averages: mean of per-seed percentile bounds
ap = pd.read_csv(ARC / "results_clean_200k/all_seed_paired.csv")
chk = ap.groupby("control_policy_id")[["delta_match_mean", "delta_match_lo", "delta_match_hi", "delta_fqe_mean", "delta_fqe_lo", "delta_fqe_hi"]].mean()
print("Mean of per-seed bootstrap means/bounds (what Table S2 reports):\n", chk.round(4).to_string(), flush=True)

# ---- 2. pooled estimator
sel = json.load(open(ARC / "results_clean_200k/selection.json"))
rows = C.read_transitions(DATA); assert K.sha256_file(DATA) == sel["dataset_sha256"]
train, val, test = K.split_fixed(rows, int(sel["split_seed"]))
sidx = C.build_state_index(rows); NA = 7; G = sel["training"]["gamma"]
arr = C.to_arrays(train, sidx)
ts = np.array([sidx[r["state"]] for r in test]); ta = np.array([r["action"] for r in test])
users, uidx = np.unique([r["user_id"] for r in test], return_inverse=True); U = len(users)
N_u = np.bincount(uidx, minlength=U).astype(float); N = N_u.sum()
seeds = [2026, 2027, 2028, 2029, 2030]
controls = ["cql_020", "cql_val_mse", "cql_final_alpha_matched", "cql_alpha_Em1"]
def per_trans(q, pi, seed):
    am = (pi[ts] == ta).astype(float)
    qe = F.fqe_fit(pi, train, sidx, G, 0.05, 60, seed, NA, arr)
    fq = C.fqe_state_values(qe, test, pi, sidx)
    return am, fq
# D[metric][control] = (S x U) matrix of per-learner sums of paired differences, per seed
D = {m: {c: np.zeros((len(seeds), U)) for c in controls} for m in ("match", "fqe")}
for si, sd in enumerate(seeds):
    z = np.load(ARC / f"results_clean_200k/seed_{sd}_q_tables.npz")
    qa = z["aptq_norm_rtg"]; pa = qa.argmax(1); am_a, fq_a = per_trans(qa, pa, sd)
    for c in controls:
        qc = np.load(Q13 / f"q_cql_alphaEm1_seed{sd}.npy") if c == "cql_alpha_Em1" else z[c]
        pc = qc.argmax(1); am_c, fq_c = per_trans(qc, pc, sd)
        D["match"][c][si] = np.bincount(uidx, weights=am_a - am_c, minlength=U)
        D["fqe"][c][si] = np.bincount(uidx, weights=fq_a - fq_c, minlength=U)
    print("seed", sd, "done", flush=True)

rng = np.random.default_rng(20261005); B = 4000
res = []
for m in ("match", "fqe"):
    for c in controls:
        M = D[m][c]                                   # S x U
        per_seed_point = M.sum(1) / N                 # point estimate for each seed
        theta = M.mean(0).sum() / N                   # seed-averaged pooled estimate
        # (a) learner bootstrap, same learner draw for every seed, seeds averaged (estimand: learner-population mean
        #     difference for the seed-averaged policy pair, training split fixed)
        th_a = np.empty(B)
        for b in range(B):
            w = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
            th_a[b] = (M.mean(0) @ w) / (N_u @ w)
        # (b) seed-to-seed variation only (training randomness): t interval over per-seed point estimates
        S = len(seeds); sd_s = per_seed_point.std(ddof=1); tcrit = 2.7764  # t_{0.975,4}
        seed_lo, seed_hi = per_seed_point.mean() - tcrit * sd_s / math.sqrt(S), per_seed_point.mean() + tcrit * sd_s / math.sqrt(S)
        # (c) two-way: resample learners AND seeds (conservative; combines both sources)
        th_c = np.empty(B)
        for b in range(B):
            w = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
            k = rng.integers(0, S, S)
            th_c[b] = (M[k].mean(0) @ w) / (N_u @ w)
        q = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
        res.append({"metric": m, "control": c, "estimate": float(theta), "seed_sd": float(sd_s),
                    "learner_boot_lo": q(th_a)[0], "learner_boot_hi": q(th_a)[1],
                    "seed_t_lo": float(seed_lo), "seed_t_hi": float(seed_hi),
                    "twoway_lo": q(th_c)[0], "twoway_hi": q(th_c)[1],
                    "per_seed_points": [float(x) for x in per_seed_point]})
df = pd.DataFrame(res); df.to_csv(OUT / "r14_pooled_intervals.csv", index=False)
json.dump(res, open(OUT / "r14_pooled_intervals.json", "w"), indent=1)
chk.to_csv(OUT / "r14_current_tableS2_basis.csv")
pd.set_option("display.width", 250)
print(df.drop(columns=["per_seed_points"]).round(4).to_string())
print("DONE")
