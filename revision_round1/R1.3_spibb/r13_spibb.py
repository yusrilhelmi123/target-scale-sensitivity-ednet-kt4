import sys, csv, json, time
from pathlib import Path
import numpy as np
from numba import njit
ARC = Path(sys.argv[1]); DATA = sys.argv[2]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
import common as C, clean_200k as K
from exp_baselines import spibb_policy

sel = json.load(open(ARC / "results_clean_200k/selection.json"))
rows = C.read_transitions(DATA); assert K.sha256_file(DATA) == sel["dataset_sha256"]
train, val, test = K.split_fixed(rows, int(sel["split_seed"]))
sidx = C.build_state_index(rows); n = len(sidx); NA = 7; G = sel["training"]["gamma"]
counts = C.state_action_counts(train, sidx, NA); beh = C.behaviour_policy_from_counts(counts)
mdp = C.build_empirical_mdp(train, sidx, NA); trans, rew_sum, sa_count, term = mdp
arr = C.to_arrays(train, sidx); test_init = K.initial_state_rows(test)
ts = np.array([sidx[r["state"]] for r in test]); ta = np.array([r["action"] for r in test])
vs = np.array([sidx[r["state"]] for r in val]); va = np.array([r["action"] for r in val])
tis = np.array([sidx[r["state"]] for r in test_init])

def spibb_stoch(counts, n_wedge, iters=60):
    """Exact copy of exp_baselines.spibb_policy, but returns the stochastic policy pi (before argmax)."""
    behaviour = C.behaviour_policy_from_counts(counts); boot = counts < n_wedge
    pi = behaviour.copy(); Q = np.zeros((n, NA))
    for _ in range(iters):
        Qn = np.zeros((n, NA))
        for s in range(n):
            for a in range(NA):
                c = sa_count[s, a]
                if c == 0: continue
                r = rew_sum[s, a] / c; bv = 0.0
                for ns, cnt in trans[(s, a)].items(): bv += (cnt / c) * float(np.sum(pi[ns] * Q[ns]))
                Qn[s, a] = r + G * bv
        Q = Qn
        npi = np.zeros((n, NA))
        for s in range(n):
            mb = float(np.sum(behaviour[s][boot[s]])); npi[s][boot[s]] = behaviour[s][boot[s]]
            free = np.where(~boot[s])[0]
            if len(free) > 0:
                best = free[int(np.argmax(Q[s][free]))]; npi[s][best] += (1.0 - mb)
            else: npi[s] = behaviour[s]
        pi = npi
    return pi, Q

def eval_mdp_stoch(pi, iters=2000, tol=1e-9):
    V = np.zeros(n)
    for _ in range(iters):
        nv = np.zeros(n); d = 0.0
        for s in range(n):
            tot = 0.0
            for a in range(NA):
                c = sa_count[s, a]
                if c == 0 or pi[s, a] == 0: continue
                bv = sum((cnt / c) * V[ns] for ns, cnt in trans[(s, a)].items())
                tot += pi[s, a] * (rew_sum[s, a] / c + G * bv)
            nv[s] = tot; d = max(d, abs(nv[s] - V[s]))
        V = nv
        if d < tol: break
    return V

@njit(cache=True)
def _fqe_epoch(qe, S, NS, A, R, D, pi, order, lr, g):
    na = qe.shape[1]
    for k in range(order.shape[0]):
        i = order[k]; t = R[i]
        if not D[i]:
            e = 0.0
            for a in range(na): e += pi[NS[i], a] * qe[NS[i], a]
            t += g * e
        qe[S[i], A[i]] += lr * (t - qe[S[i], A[i]])

def fqe_stoch(pi, seed, epochs=60, lr=0.05):
    qe = np.zeros((n, NA)); S, NS, A, R, D = arr["s"], arr["ns"], arr["a"], arr["r"], arr["done"]
    order = np.arange(len(S))
    for ep in range(epochs):
        np.random.default_rng(seed + ep).shuffle(order); _fqe_epoch(qe, S, NS, A, R, D, pi, order, lr, G)
    return qe

def fqe_vals(pi, seeds=(2026, 2027, 2028, 2029, 2030)):
    v, vi = [], []
    for sd in seeds:
        qe = fqe_stoch(pi, sd); ev = (pi * qe).sum(1)
        v.append(float(ev[ts].mean())); vi.append(float(ev[tis].mean()))
    return np.mean(v), np.std(v, ddof=1), np.mean(vi)

out = []
for nw in [2, 5, 10, 20, 50]:
    pi_s, Q = spibb_stoch(counts, nw)
    pi_a, Q_arc = spibb_policy(counts, *mdp, n, NA, G, n_wedge=nw)
    assert (pi_s.argmax(1) == pi_a).all() and np.abs(Q - Q_arc).max() < 1e-12, "stoch copy != archived"
    pi_d = np.eye(NA)[pi_a]
    Vs = eval_mdp_stoch(pi_s); Vd = C.evaluate_policy_empirical_mdp(pi_a, *mdp, n, G)
    boot = counts < nw
    # determinisation diagnostics: states where argmax picks a bootstrapped (behaviour-pinned) action
    picks_boot = boot[np.arange(n), pi_a]
    mass_boot = (pi_s * boot).sum(1)
    fs, fs_sd, fs_i = fqe_vals(pi_s); fd, fd_sd, fd_i = fqe_vals(pi_d)
    r = {"n_wedge": nw,
         "val_emp_value_det": float(Vd[vs].mean()), "val_emp_value_stoch": float(Vs[vs].mean()),
         "val_action_match_det": float((pi_a[vs] == va).mean()), "val_action_match_stoch": float(pi_s[vs, va].mean()),
         "test_action_match_det": float((pi_a[ts] == ta).mean()), "test_action_match_stoch": float(pi_s[ts, ta].mean()),
         "test_fqe_det": fd, "test_fqe_det_sd": fd_sd, "test_fqe_stoch": fs, "test_fqe_stoch_sd": fs_sd,
         "test_fqe_init_det": fd_i, "test_fqe_init_stoch": fs_i,
         "test_emp_value_det": float(Vd[ts].mean()), "test_emp_value_stoch": float(Vs[ts].mean()),
         "frac_states_boot_mass_gt0": float((mass_boot > 0).mean()),
         "test_w_mean_mass_on_bootstrapped": float(mass_boot[ts].mean()),
         "frac_states_argmax_on_bootstrapped": float(picks_boot.mean()),
         "test_w_frac_argmax_on_bootstrapped": float(picks_boot[ts].mean()),
         "n_bootstrapped_pairs": int(boot.sum()), "n_pairs_total": int(n * NA)}
    out.append(r); print(nw, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
# behaviour policy reference (stochastic baseline of SPIBB)
fb, fb_sd, fb_i = fqe_vals(beh); Vb = eval_mdp_stoch(beh)
ref = {"behaviour_test_fqe": fb, "behaviour_test_fqe_sd": fb_sd, "behaviour_test_fqe_init": fb_i,
       "behaviour_test_emp_value": float(Vb[ts].mean()), "behaviour_val_emp_value": float(Vb[vs].mean()),
       "behaviour_test_action_match": float(beh[ts, ta].mean())}
print(ref)
json.dump({"grid": out, "behaviour_reference": ref}, open(OUT / "r13_spibb.json", "w"), indent=1)
with open(OUT / "r13_spibb_grid.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("DONE")
