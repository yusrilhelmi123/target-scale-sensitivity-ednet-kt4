"""v1.1 diagnostic: why do single runs show large APTQ-vs-matched-CQL action-match gaps?
For each (split, seed) it retrains the APTQ-inspired scaled-RTG policy and its final-alpha-matched
CQL control (fastrl, identical to common.train_q), lists the states where the greedy actions differ,
their share of test transitions, and the gap between the two best Q-values in each policy.
Usage: python near_tie_check.py DATA.csv OUT.json"""
import sys, json
import numpy as np
import common as C, fastrl as F
DATA, OUT = sys.argv[1], sys.argv[2]
rows = C.read_transitions(DATA); sidx = C.build_state_index(rows)
res = []
for sp, sd in [(5678, 2026), (9012, 2030), (20260728, 2026)]:
    tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, sp); arr = C.to_arrays(tr, sidx)
    q, h = F.train_q(tr, sidx, 7, algorithm="aptq", aptq_target_scale=0.10, aptq_target_quantile=0.90, seed=sd, arrays=arr)
    fa = h[-1]["alpha"]
    qm, _ = F.train_q(tr, sidx, 7, algorithm="cql", cql_alpha=fa, seed=sd, arrays=arr)
    pa, pm = C.greedy_policy(q), C.greedy_policy(qm)
    ts = np.array([sidx[r["state"]] for r in te]); ta = np.array([r["action"] for r in te])
    diff = np.where(pa != pm)[0]
    states = []
    for s in diff:
        share = float((ts == s).mean())
        if share > 0:
            g = lambda Q: float(np.sort(Q[s])[-1] - np.sort(Q[s])[-2])
            states.append({"state": int(s), "test_share": share, "top2_gap_aptq": g(q), "top2_gap_cql": g(qm),
                           "aptq_action": int(pa[s]), "cql_action": int(pm[s])})
    states.sort(key=lambda x: -x["test_share"])
    rec = {"split": sp, "seed": sd, "final_alpha": fa,
           "action_match_aptq": float((pa[ts] == ta).mean()), "action_match_cql": float((pm[ts] == ta).mean()),
           "n_differing_states": int(len(diff)), "test_share_differing_states": float(np.isin(ts, diff).mean()),
           "differing_states": states}
    res.append(rec)
    top = states[0] if states else None
    print(sp, sd, "match %.3f vs %.3f" % (rec["action_match_aptq"], rec["action_match_cql"]),
          "diff states", len(diff), "share %.3f" % rec["test_share_differing_states"],
          "top state", top and {k: round(v, 4) if isinstance(v, float) else v for k, v in top.items()})
json.dump(res, open(OUT, "w"), indent=1)
