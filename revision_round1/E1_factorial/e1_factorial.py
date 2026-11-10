import sys, csv, json, time, itertools
from pathlib import Path
import numpy as np
ARC = Path(sys.argv[1]); DATA = sys.argv[2]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
import common as C, fastrl as F, clean_200k as K

sel = json.load(open(ARC / "results_clean_200k/selection.json"))
rows = C.read_transitions(DATA); assert K.sha256_file(DATA) == sel["dataset_sha256"]
train, val, test = K.split_fixed(rows, int(sel["split_seed"]))
sidx = C.build_state_index(rows); NA = 7; G = sel["training"]["gamma"]; LR = sel["training"]["learning_rate"]; EP = sel["training"]["epochs"]
arr = C.to_arrays(train, sidx); va = C.to_arrays(val, sidx); te = C.to_arrays(test, sidx)
S = len(sidx); ts = te["s"]; ta = te["a"]
tv = np.bincount(ts, minlength=S) / len(ts)
def val_mse(q):
    tgt = va["r"] + (1 - va["done"]) * G * q[va["ns"]].max(1)
    return float(np.mean((tgt - q[va["s"], va["a"]]) ** 2))
def test_am(pi): return float(np.mean(pi[ts] == ta))
def fqe_test(pi, seed):
    qe = F.fqe_fit(pi, train, sidx, G, 0.05, 60, seed, NA, arr)
    return float(C.fqe_state_values(qe, test, pi, sidx).mean())

SEEDS = [2026, 2027, 2028, 2029, 2030]
SCALES = [0.05, 0.10, 0.20, 0.50]; ETAS = [0.005, 0.02, 0.08]; A0S = [0.05, 0.20, 0.50]
CQL_GRID = list(np.geomspace(0.01, 1.0, 36))
t0 = time.time()
# ---- fixed CQL comparator (same budget: 36 configurations)
cql_rows = []; best_cql = {}
CACHE = OUT / "e1_cql_best.json"
if CACHE.exists():
    best_cql = {int(k): tuple(v) for k, v in json.load(open(CACHE)).items()}
for sd in ([] if best_cql else SEEDS):
    best = None
    for a in CQL_GRID:
        q, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=float(a), epochs=EP, lr=LR, gamma=G, seed=sd, arrays=arr)
        m = val_mse(q); cql_rows.append({"seed": sd, "alpha": float(a), "val_mse": m, "val_am": float(np.mean(q.argmax(1)[va["s"]] == va["a"]))})
        if best is None or m < best[1]: best = (float(a), m, q)
    best_cql[sd] = best
    print(f"CQL seed {sd}: best alpha={best[0]:.4f} val_mse={best[1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
if cql_rows:
    json.dump(cql_rows, open(OUT / "e1_cql_budget_grid.json", "w"))
    json.dump({str(k): [v[0], v[1]] for k, v in best_cql.items()}, open(CACHE, "w"))

# ---- adaptive runs
RUNS = OUT / "e1_runs.json"
runs = json.load(open(RUNS)) if RUNS.exists() else []
done = {(r["scale"], r["eta"], r["alpha0"]) for r in runs}
MAXC = int(sys.argv[5]) if len(sys.argv) > 5 else 99   # max configurations this call
newc = 0
for sc, eta, a0 in itertools.product(SCALES, ETAS, A0S):
    if (sc, eta, a0) in done: continue
    if newc >= MAXC: break
    newc += 1
    for sd in SEEDS:
        q, h = F.train_q(train, sidx, NA, algorithm="aptq", epochs=EP, lr=LR, gamma=G, aptq_initial_alpha=a0, aptq_alpha_lr=eta,
                         aptq_target_source="return_to_go_quantile", aptq_target_transform="scale", aptq_target_scale=sc,
                         aptq_target_quantile=0.90, seed=sd, arrays=arr)
        al = [x["alpha"] for x in h]; a_E, a_Em1 = al[-1], al[-2]
        m = val_mse(q); ref = best_cql[sd][1]
        c_collapse = a_E < 0.01; c_infl = max(al) > 1.0; c_mse = m > 1.10 * ref
        stable = not (c_collapse or c_infl or c_mse)
        qm, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=float(a_Em1), epochs=EP, lr=LR, gamma=G, seed=sd, arrays=arr)
        pa, pm = q.argmax(1), qm.argmax(1)
        r = {"scale": sc, "eta": eta, "alpha0": a0, "seed": sd, "alpha_E": a_E, "alpha_Em1": a_Em1, "alpha_max": max(al), "alpha_min": min(al),
             "alpha_traj": [round(x, 6) for x in al], "val_mse": m, "best_cql_val_mse": ref, "mse_ratio": m / ref,
             "val_am": float(np.mean(pa[va["s"]] == va["a"])), "collapse": c_collapse, "inflation": c_infl, "mse_gap": c_mse, "stable": stable,
             "test_am_apt": test_am(pa), "test_am_matched": test_am(pm), "disagree_test_w": float((tv * (pa != pm)).sum())}
        if stable:  # FQE only for stable runs (descriptive comparison with matched control)
            r["test_fqe_apt"] = fqe_test(pa, sd); r["test_fqe_matched"] = fqe_test(pm, sd)
        runs.append(r)
    ns = sum(x["stable"] for x in runs[-5:])
    print(f"scale={sc} eta={eta} a0={a0}: stable {ns}/5  aE={[round(x['alpha_E'],3) for x in runs[-5:]]}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(runs, open(RUNS, "w"))
print("CHUNK_DONE configs_completed=%d/36" % len({(r["scale"], r["eta"], r["alpha0"]) for r in runs}), flush=True)
