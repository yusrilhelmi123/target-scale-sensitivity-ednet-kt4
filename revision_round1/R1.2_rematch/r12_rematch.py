import sys, csv, json, time
from pathlib import Path
import numpy as np

ARC = Path(sys.argv[1]); DATA = sys.argv[2]; OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARC / "scripts"))
import common as C, fastrl as F, clean_200k as K

sel = json.load(open(ARC / "results_clean_200k/selection.json"))
rows = C.read_transitions(DATA)
assert K.sha256_file(DATA) == sel["dataset_sha256"]
train, val, test = K.split_fixed(rows, int(sel["split_seed"]))
sidx = C.build_state_index(rows)
NA = 7; G = sel["training"]["gamma"]; LR = sel["training"]["learning_rate"]; EP = sel["training"]["epochs"]
arr = C.to_arrays(train, sidx)
test_init = K.initial_state_rows(test)
S = len(sidx)
ts = np.array([sidx[r["state"]] for r in test]); test_visit = np.bincount(ts, minlength=S) / len(ts)
train_visit = np.bincount(arr["s"], minlength=S) / len(arr["s"])
tact = np.array([r["action"] for r in test])
def action_match(pi): return float(np.mean(pi[ts] == tact))

def compare(qa, qc):
    pa, pc = qa.argmax(1), qc.argmax(1)
    dis = pa != pc
    out = {"n_states_disagree": int(dis.sum()), "disagree_unweighted": float(dis.mean()),
           "disagree_test_weighted": float((test_visit * dis).sum()),
           "disagree_train_weighted": float((train_visit * dis).sum())}
    idx = np.arange(S)
    gap_a = qa[idx, pa] - qa[idx, pc]
    gap_c = qc[idx, pc] - qc[idx, pa]
    out["qgap_apt_testw_all"] = float((test_visit * gap_a).sum())
    out["qgap_ctl_testw_all"] = float((test_visit * gap_c).sum())
    w = test_visit * dis
    if w.sum() > 0:
        out["qgap_apt_mean_at_disagree_testw"] = float((w * gap_a).sum() / w.sum())
        out["qgap_ctl_mean_at_disagree_testw"] = float((w * gap_c).sum() / w.sum())
        out["qgap_apt_max"] = float(gap_a[dis].max()); out["qgap_ctl_max"] = float(gap_c[dis].max())
        srt = np.sort(qa, axis=1); top2 = srt[:, -1] - srt[:, -2]
        out["apt_top2gap_median_at_disagree"] = float(np.median(top2[dis]))
    out["q_absdiff_testw"] = float((test_visit * np.abs(qa - qc).mean(1)).sum())
    return out

res = []; t0 = time.time()
for seed in [2026, 2027, 2028, 2029, 2030]:
    z = np.load(ARC / f"results_clean_200k/seed_{seed}_q_tables.npz")
    q_apt, q_old = z["aptq_norm_rtg"], z["cql_final_alpha_matched"]
    h = {int(r["epoch"]): float(r["alpha"]) for r in csv.DictReader(open(ARC / f"results_clean_200k/seed_{seed}_histories.csv"))
         if r["policy_id"] == "aptq_norm_rtg"}
    a_E, a_Em1 = h[EP], h[EP - 1]
    q_rep, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=a_E, epochs=EP, lr=LR, gamma=G, seed=seed, arrays=arr)
    rep_old = float(np.abs(q_rep - q_old).max())
    q_new, _ = F.train_q(train, sidx, NA, algorithm="cql", cql_alpha=a_Em1, epochs=EP, lr=LR, gamma=G, seed=seed, arrays=arr)
    np.save(OUT / f"q_cql_alphaEm1_seed{seed}.npy", q_new)
    ap = sel["aptq_norm_rtg"]
    q_ap2, h2 = F.train_q(train, sidx, NA, algorithm="aptq", epochs=EP, lr=LR, gamma=G, aptq_initial_alpha=ap["initial_alpha"],
                          aptq_alpha_lr=ap["eta"], aptq_target_source="return_to_go_quantile", aptq_target_transform="scale",
                          aptq_target_scale=ap["scale"], aptq_target_quantile=ap["quantile"], seed=seed, arrays=arr)
    rep_apt = float(np.abs(q_ap2 - q_apt).max())
    pa, po, pn = q_apt.argmax(1), q_old.argmax(1), q_new.argmax(1)
    fq = {}
    for name, pi in (("apt", pa), ("old", po), ("new", pn)):
        qe = F.fqe_fit(pi, train, sidx, G, 0.05, 60, seed, NA, arr)
        fq[name] = float(C.fqe_state_values(qe, test, pi, sidx).mean())
        fq[name + "_init"] = float(C.fqe_state_values(qe, test_init, pi, sidx).mean())
    r = {"seed": seed, "alpha_E": a_E, "alpha_Em1": a_Em1, "alpha_diff": a_E - a_Em1,
         "repro_maxabs_old_control_Q": rep_old, "repro_maxabs_aptq_Q": rep_apt,
         "am_apt": action_match(pa), "am_old": action_match(po), "am_new": action_match(pn),
         "fqe_apt": fq["apt"], "fqe_old": fq["old"], "fqe_new": fq["new"],
         "fqe_init_apt": fq["apt_init"], "fqe_init_old": fq["old_init"], "fqe_init_new": fq["new_init"],
         "same_policy_old_vs_new": bool((po == pn).all())}
    for pref, qc in (("old", q_old), ("new", q_new)):
        for k, v in compare(q_apt, qc).items(): r[f"{pref}_{k}"] = v
    res.append(r)
    print(f"seed {seed} {time.time()-t0:.0f}s repro old={rep_old:.2e} apt={rep_apt:.2e} aE={a_E:.6f} aEm1={a_Em1:.6f} "
          f"dis_old={r['old_disagree_test_weighted']:.4f} dis_new={r['new_disagree_test_weighted']:.4f}", flush=True)
    json.dump(res, open(OUT / "r12_results.json", "w"), indent=1)
keys = sorted({k for r in res for k in r}, key=lambda k: list(res[0]).index(k) if k in res[0] else 999)
with open(OUT / "r12_per_seed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
print("DONE", flush=True)
