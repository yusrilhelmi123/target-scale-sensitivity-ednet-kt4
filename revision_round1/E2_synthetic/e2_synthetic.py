import sys, json, time, random, bisect
from pathlib import Path
import numpy as np
ARC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); BI = int(sys.argv[3]); MAXW = int(sys.argv[4]); NW = int(sys.argv[5]) if len(sys.argv) > 5 else 20
sys.path.insert(0, str(ARC / "scripts"))
import common as C, fastrl as F
S, A, G, GR, PTERM = 600, 7, 0.95, 0.99, 0.02
BETAS = [0.5, 1.5, 3.0]; EP, LR = 20, 0.08
OUT.mkdir(parents=True, exist_ok=True); FN = OUT / f"e2_worlds_beta{BETAS[BI]}.jsonl"
sidx = {f"s{i}": i for i in range(S)}

def make_world(ws, beta):
    rng = np.random.default_rng(ws)
    rho = 1.0 / np.arange(1, S + 1); rho = rho[rng.permutation(S)]; rho /= rho.sum()
    local = rng.integers(0, S, size=(S, A, 3))
    u = rng.normal(0.2, 0.5, S); v = rng.normal(0, 0.6, (S, A)); p = 1 / (1 + np.exp(-(u[:, None] + v))); rbar = 2 * p - 1
    w = rng.normal(0, 1, (S, A)); e = np.exp(beta * w); pib = e / e.sum(1, keepdims=True)
    P = np.zeros((S, A, S)); P += 0.5 * rho[None, None, :]
    for k in range(3): np.add.at(P, (np.arange(S)[:, None], np.arange(A)[None, :], local[:, :, k]), 0.5 / 3)
    return dict(rho=rho, local=local, p=p, rbar=rbar, pib=pib, P=P)

def simulate(W, n_ep, seed):
    rnd = random.Random(seed); cum_pib = np.cumsum(W["pib"], 1).tolist(); cum_rho = np.cumsum(W["rho"]).tolist(); p = W["p"].tolist(); local = W["local"].tolist()
    rows = []
    for ep in range(n_ep):
        s = min(bisect.bisect(cum_rho, rnd.random()), S - 1); traj = []
        while True:
            a = min(bisect.bisect(cum_pib[s], rnd.random()), A - 1)
            r = 1.0 if rnd.random() < p[s][a] else -1.0
            done = 1 if rnd.random() < PTERM else 0
            if done: ns = s
            elif rnd.random() < 0.5: ns = min(bisect.bisect(cum_rho, rnd.random()), S - 1)
            else: ns = local[s][a][rnd.randrange(3)]
            traj.append([s, a, r, done, ns])
            if done: break
            s = ns
        g = 0.0; rtg = [0.0] * len(traj)
        for i in range(len(traj) - 1, -1, -1):
            g = traj[i][2] + GR * g * (0 if traj[i][3] else 1); rtg[i] = g
        for i, (s_, a_, r_, d_, ns_) in enumerate(traj):
            rows.append({"user_id": f"u{seed}_{ep}", "state": f"s{s_}", "next_state": f"s{ns_}", "action": a_, "reward": r_, "done": d_, "return_to_go": rtg[i]})
    return rows

def true_values(W, pi):
    """pi: deterministic (S,) or stochastic (S,A). Returns V (S,)."""
    if pi.ndim == 1:
        r = W["rbar"][np.arange(S), pi]; Ppi = W["P"][np.arange(S), pi]
    else:
        r = (pi * W["rbar"]).sum(1); Ppi = np.einsum("sa,sat->st", pi, W["P"])
    return np.linalg.solve(np.eye(S) - G * (1 - PTERM) * Ppi, r)

def run_world(beta, ws):
    W = make_world(ws, beta)
    tr = simulate(W, 2600, ws * 3 + 1); va = simulate(W, 560, ws * 3 + 2); te = simulate(W, 560, ws * 3 + 3)
    arr = C.to_arrays(tr, sidx); vv = C.to_arrays(va, sidx); tt = C.to_arrays(te, sidx)
    tv = np.bincount(tt["s"], minlength=S) / len(tt["s"])
    counts = C.state_action_counts(tr, sidx, A)
    mdp = C.build_empirical_mdp(tr, sidx, A)
    def vmse(q):
        tg = vv["r"] + (1 - vv["done"]) * G * q[vv["ns"]].max(1); return float(np.mean((tg - q[vv["s"], vv["a"]]) ** 2))
    def am(pi): return float(np.mean(pi[tt["s"]] == tt["a"]))
    # learners
    q_off, _ = F.train_q(tr, sidx, A, algorithm="offline_q", epochs=EP, lr=LR, gamma=G, seed=ws, arrays=arr)
    best = None
    for al in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        q, _ = F.train_q(tr, sidx, A, algorithm="cql", cql_alpha=al, epochs=EP, lr=LR, gamma=G, seed=ws, arrays=arr); m = vmse(q)
        if best is None or m < best[0]: best = (m, al, q)
    q_cql, a_cql = best[2], best[1]
    bapt = None
    for sc in [0.05, 0.1, 0.2, 0.5]:
        for qu in [0.6, 0.8, 0.9]:
            q, h = F.train_q(tr, sidx, A, algorithm="aptq", epochs=EP, lr=LR, gamma=G, aptq_initial_alpha=0.2, aptq_alpha_lr=0.02,
                             aptq_target_source="return_to_go_quantile", aptq_target_transform="scale", aptq_target_scale=sc, aptq_target_quantile=qu, seed=ws, arrays=arr); m = vmse(q)
            if bapt is None or m < bapt[0]: bapt = (m, sc, qu, q, h)
    q_apt, h_apt = bapt[3], bapt[4]; a_E, a_Em1 = h_apt[-1]["alpha"], h_apt[-2]["alpha"]
    q_mat, _ = F.train_q(tr, sidx, A, algorithm="cql", cql_alpha=float(a_Em1), epochs=EP, lr=LR, gamma=G, seed=ws, arrays=arr)
    Vb = true_values(W, W["pib"])
    out = {"beta": beta, "world": ws, "n_train": len(tr), "frac_pairs_lt2": float((counts < 2).mean()), "frac_pairs_zero": float((counts == 0).mean()),
           "true_behaviour_vis": float(tv @ Vb), "true_behaviour_init": float(W["rho"] @ Vb), "apt_scale": bapt[1], "apt_quantile": bapt[2], "cql_alpha": a_cql,
           "alpha_E": a_E, "alpha_Em1": a_Em1}
    pols = {"offline": q_off, "cql": q_cql, "aptq": q_apt, "matched": q_mat}
    for name, q in pols.items():
        pi = q.argmax(1); V = true_values(W, pi)
        qe = F.fqe_fit(pi, tr, sidx, G, 0.05, 60, ws, A, arr); Ve = C.evaluate_policy_empirical_mdp(pi, *mdp, S, G)
        out[name] = {"own": float(tv @ q.max(1)), "fqe": float(tv @ qe[np.arange(S), pi]), "emp": float(tv @ Ve), "true": float(tv @ V),
                     "true_init": float(W["rho"] @ V), "action_match": am(pi), "val_mse": vmse(q)}
    pa, pm = q_apt.argmax(1), q_mat.argmax(1)
    out["disagree_apt_matched_test_w"] = float((tv * (pa != pm)).sum())
    return out

done = set()
if FN.exists():
    for ln in open(FN): done.add(json.loads(ln)["world"])
t0 = time.time(); n = 0
for k in range(NW):
    ws = 1000 * (BI + 1) + k
    if ws in done: continue
    if n >= MAXW: break
    r = run_world(BETAS[BI], ws); n += 1
    with open(FN, "a") as f: f.write(json.dumps(r) + "\n")
    print(f"beta={BETAS[BI]} world={ws} [{time.time()-t0:.0f}s] true_beh={r['true_behaviour_vis']:.3f} "
          f"offline own/fqe/emp/true={r['offline']['own']:.2f}/{r['offline']['fqe']:.2f}/{r['offline']['emp']:.2f}/{r['offline']['true']:.2f} "
          f"aptq true={r['aptq']['true']:.3f} matched true={r['matched']['true']:.3f} frac<2={r['frac_pairs_lt2']:.2f}", flush=True)
print("CHUNK_DONE", BI, len(done) + n, flush=True)
