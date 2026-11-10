"""Check fastrl against archived v1.0 Q-tables and FQE values (seed 2026)."""
import sys, time, json, numpy as np, pandas as pd
import common as C, fastrl as F
D = sys.argv[1]; OLD = sys.argv[2]
rows = C.read_transitions(D); sidx = C.build_state_index(rows)
tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, 20260728)
arr = C.to_arrays(tr, sidx)
old = np.load(f"{OLD}/seed_2026_q_tables.npz", allow_pickle=True)
t = time.time()
q, h = F.train_q(tr, sidx, 7, algorithm="aptq", aptq_target_scale=0.10, aptq_target_quantile=0.90, seed=2026, arrays=arr)
print("aptq max|dQ| vs archived:", np.abs(q - old["aptq_norm_rtg"]).max(), "final alpha", h[-1]["alpha"], "t", time.time() - t)
qc, _ = F.train_q(tr, sidx, 7, algorithm="cql", cql_alpha=h[-1]["alpha"], seed=2026, arrays=arr)
print("cql matched max|dQ|:", np.abs(qc - old["cql_final_alpha_matched"]).max())
m = pd.read_csv(f"{OLD}/seed_2026_metrics.csv")
t = time.time()
v = F.fqe_value(C.greedy_policy(q), tr, te, sidx, 0.95, seed=2026, arrays=arr)
print("fqe", v, "archived", float(m[m.policy_id == "aptq_norm_rtg"].fqe_value_visitation.iloc[0]), "t", time.time() - t)
