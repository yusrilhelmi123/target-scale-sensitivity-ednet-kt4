"""v1.1 long-horizon check: adaptive-coefficient trajectories over 100 epochs
(fixed split 20260728, training seeds 2026-2030) for the scaled-RTG,
reward-quantile and moving dataset-Q targets. Q-learning settings otherwise
identical to the main protocol (lr 0.08, gamma 0.95, eta 0.02, alpha0 0.20)."""
import sys, json
from pathlib import Path
import common as C, fastrl as F
DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
rows = C.read_transitions(DATA); sidx = C.build_state_index(rows)
tr, va, te = C.split_by_user_three_way(rows, 0.15, 0.15, 20260728); arr = C.to_arrays(tr, sidx)
T = {"scaled_rtg": dict(aptq_target_source="return_to_go_quantile", aptq_target_scale=0.10, aptq_target_quantile=0.90),
     "reward_quantile": dict(aptq_target_source="reward_quantile", aptq_target_scale=0.10, aptq_target_quantile=0.60),
     "moving_dataset_q": dict(aptq_target_source="dataset_q_quantile", aptq_target_quantile=0.80)}
with open(OUT / "long_horizon.jsonl", "w") as f:
    for name, kw in T.items():
        for seed in [2026, 2027, 2028, 2029, 2030]:
            q, h = F.train_q(tr, sidx, 7, algorithm="aptq", epochs=100, seed=seed, arrays=arr, **kw)
            f.write(json.dumps({"target": name, "seed": seed, "history": [
                {"epoch": x["epoch"], "alpha": x["alpha"], "target_q": x["target_q"], "mean_dataset_q": x["mean_dataset_q"]} for x in h]}) + "\n")
            print(name, seed, round(h[19]["alpha"], 4), round(h[-1]["alpha"], 4), flush=True)
