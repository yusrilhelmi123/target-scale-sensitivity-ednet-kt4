import sys, re, json, csv, time
from pathlib import Path
import numpy as np
ARC = Path(sys.argv[1]); DATA = sys.argv[2]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
import common as C, fastrl as F, clean_200k as K
sel = json.load(open(ARC / "results_clean_200k/selection.json"))
rows = C.read_transitions(DATA); assert K.sha256_file(DATA) == sel["dataset_sha256"]
strip = lambda s: re.sub(r"_g\d", "", s)
for r in rows: r["state"] = strip(r["state"]); r["next_state"] = strip(r["next_state"])
train, val, test = K.split_fixed(rows, int(sel["split_seed"]))
sidx = C.build_state_index(rows); S = len(sidx); NA = 7; G = 0.95; LR = 0.08; EP = 20
print("states without gap bin:", S, flush=True)
arr = C.to_arrays(train, sidx); va = C.to_arrays(val, sidx); te = C.to_arrays(test, sidx); ts, ta = te["s"], te["a"]
tv = np.bincount(ts, minlength=S) / len(ts)
def vmse(q):
    t = va["r"] + (1 - va["done"]) * G * q[va["ns"]].max(1); return float(np.mean((t - q[va["s"], va["a"]]) ** 2))
am = lambda pi: float(np.mean(pi[ts] == ta))
def fqe(pi, sd):
    qe = F.fqe_fit(pi, train, sidx, G, 0.05, 60, sd, NA, arr); return float(C.fqe_state_values(qe, test, pi, sidx).mean())
# selection on validation (seed 2026), same protocol as the main study
bc = min(((vmse(F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=float(a), epochs=EP, lr=LR, gamma=G, seed=2026, arrays=arr)[0]), float(a)) for a in np.geomspace(0.01, 1.0, 12)))
ba = None
for sc in [0.05, 0.1, 0.2, 0.5]:
    for qu in [0.6, 0.8, 0.9]:
        q, h = F.train_q(train, sidx, NA, algorithm="aptq", epochs=EP, lr=LR, gamma=G, aptq_initial_alpha=0.2, aptq_alpha_lr=0.02, aptq_target_source="return_to_go_quantile",
                         aptq_target_transform="scale", aptq_target_scale=sc, aptq_target_quantile=qu, seed=2026, arrays=arr); m = vmse(q)
        if ba is None or m < ba[0]: ba = (m, sc, qu)
print(f"selected: CQL alpha={bc[1]:.4f} (val mse {bc[0]:.4f}); APTQ scale={ba[1]} q={ba[2]} (val mse {ba[0]:.4f})", flush=True)
res = []
for sd in [2026, 2027, 2028, 2029, 2030]:
    qa, h = F.train_q(train, sidx, NA, algorithm="aptq", epochs=EP, lr=LR, gamma=G, aptq_initial_alpha=0.2, aptq_alpha_lr=0.02, aptq_target_source="return_to_go_quantile",
                      aptq_target_transform="scale", aptq_target_scale=ba[1], aptq_target_quantile=ba[2], seed=sd, arrays=arr)
    aE, aEm1 = h[-1]["alpha"], h[-2]["alpha"]
    qm, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=float(aEm1), epochs=EP, lr=LR, gamma=G, seed=sd, arrays=arr)
    qc, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=bc[1], epochs=EP, lr=LR, gamma=G, seed=sd, arrays=arr)
    qo, _ = F.train_q(train, sidx, NA, algorithm="offline_q", epochs=EP, lr=LR, gamma=G, seed=sd, arrays=arr)
    pa, pm, pc, po = [q.argmax(1) for q in (qa, qm, qc, qo)]
    r = {"seed": sd, "alpha_E": aE, "alpha_Em1": aEm1, "am_apt": am(pa), "am_matched": am(pm), "am_cql": am(pc), "am_offline": am(po),
         "fqe_apt": fqe(pa, sd), "fqe_matched": fqe(pm, sd), "fqe_cql": fqe(pc, sd), "fqe_offline": fqe(po, sd),
         "disagree_apt_matched": float((tv * (pa != pm)).sum())}
    res.append(r); print(sd, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump({"states": S, "cql_alpha": bc[1], "apt_scale": ba[1], "apt_quantile": ba[2], "runs": res}, open(OUT / "r11_nogap.json", "w"), indent=1)
a = np.array
d_am = a([r["am_apt"] - r["am_matched"] for r in res]); d_f = a([r["fqe_apt"] - r["fqe_matched"] for r in res])
print("\nNO-GAP: APTQ - matched CQL: d action match mean %.4f (seeds %s) | d FQE mean %.4f (seeds %s) | disagreement %.4f" % (d_am.mean(), d_am.round(4), d_f.mean(), d_f.round(4), np.mean([r["disagree_apt_matched"] for r in res])))
print("action match mean: apt %.4f matched %.4f cql %.4f offline %.4f | FQE: apt %.3f matched %.3f cql %.3f offline %.3f" % tuple(np.mean([r[k] for r in res]) for k in ("am_apt","am_matched","am_cql","am_offline","fqe_apt","fqe_matched","fqe_cql","fqe_offline")))
