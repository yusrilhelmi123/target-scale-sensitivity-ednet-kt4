# -*- coding: utf-8 -*-
"""Regenerate Figures 2, 3, and S1 from the clean-200k results directory.
Portable: pass --results-dir (folder with summary_metrics.csv, selection_grid.csv,
seed_<seed>_histories.csv) and --out-dir. Run: python make_figures.py --results-dir results_clean_200k --out-dir figures
"""
import os, csv, argparse, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def load_metrics(d):
    return {r["policy_id"]: r for r in csv.DictReader(open(os.path.join(d, "summary_metrics.csv")))}

def fig2(d, out):
    m = load_metrics(d)
    def g(pid, base):
        r = m[pid]; mean = float(r[base + "_mean"]); lo = float(r[base + "_ci95_lo"]); hi = float(r[base + "_ci95_hi"])
        return mean, (hi - lo) / 2
    order = [("offline_q", "Offline Q", "#888888", "o"), ("cql_020", "CQL α=0.20", "#1f77b4", "s"),
             ("cql_val_mse", "CQL val-selected", "#2ca02c", "^"), ("cql_final_alpha_matched", "CQL final-α matched", "#9467bd", "D"),
             ("aptq_norm_rtg", "APTQ scaled RTG", "#d62728", "*")]
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for pid, name, c, mk in order:
        x, xe = g(pid, "action_match"); y, ye = g(pid, "fqe_value_visitation")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, color=c, ms=15 if mk == "*" else 9,
                    capsize=3, elinewidth=1, mec="k", mew=0.5, label=name, zorder=3)
    ax.add_patch(Ellipse((0.529, 4.66), 0.055, 1.15, fill=False, ls="--", ec="0.4", lw=1.0))
    ax.set_xlabel("Action match"); ax.set_ylabel("FQE value (visitation)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper center", ncol=2, fontsize=8.2)
    ax.set_title("Value–behaviour trade-off (200k test, five-seed means ± 95% CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure2_tradeoff.png"), dpi=300); plt.close(fig)

def hist_rows(d, seed):
    return list(csv.DictReader(open(os.path.join(d, f"seed_{seed}_histories.csv"))))

def seeds_available(d):
    return sorted(int(os.path.basename(f).split("_")[1]) for f in glob.glob(os.path.join(d, "seed_*_histories.csv")))

def traj(rows, pid):
    r = sorted([x for x in rows if x["policy_id"] == pid], key=lambda x: int(x["epoch"]))
    return [int(x["epoch"]) for x in r], [float(x["alpha"]) for x in r]

def fig3(d, out):
    seeds = seeds_available(d); h = hist_rows(d, seeds[0])
    series = [("aptq_norm_rtg", "Scaled RTG → α≈0.05", "#d62728", "-"),
              ("aptq_reward_q", "Reward-quantile → α>1.5", "#1f77b4", "--"),
              ("aptq_moving_dataset_q", "Moving dataset-Q → 0", "#2ca02c", "-."),
              ("aptq_fixed_reference_q", "Fixed reference-Q → 0", "#9467bd", ":")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.02))
    for pid, lab, c, ls in series:
        e, al = traj(h, pid); a1.plot(e, al, ls, color=c, lw=1.8, label=lab, marker="o", ms=3)
    a1.set_xlabel("Training epoch"); a1.set_ylabel("Coefficient α"); a1.set_xlim(1, 20)
    a1.set_title("(a) Coefficient trajectory by target source", fontsize=10); a1.legend(fontsize=8); a1.grid(True, ls=":", lw=0.5, alpha=0.6)
    grid = [r for r in csv.DictReader(open(os.path.join(d, "selection_grid.csv"))) if r["family"] == "aptq_norm_rtg" and r["scale"] in ("0.05", "0.1", "0.2")]
    by = defaultdict(list)
    for r in grid: by[float(r["scale"])].append((float(r["quantile"]), float(r["bellman_mse_val"])))
    cols = {0.05: "#1f77b4", 0.1: "#d62728", 0.2: "#2ca02c"}
    for sc in sorted(by):
        pts = sorted(by[sc]); a2.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=cols[sc], lw=1.6, ms=5, label=f"scale={sc:g}")
    a2.scatter([0.9], [1.8552], s=180, marker="*", color="gold", ec="k", lw=1, zorder=5)
    a2.set_xlabel("Target quantile"); a2.set_ylabel("Validation Bellman MSE"); a2.set_xticks([0.6, 0.8, 0.9])
    a2.set_title("(b) Sensitivity landscape (scale × quantile)", fontsize=10); a2.legend(fontsize=8.5); a2.grid(True, ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure3_target.png"), dpi=300); plt.close(fig)

def figS1(d, out):
    seeds = seeds_available(d)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.cm.viridis
    for i, s in enumerate(seeds):
        e, al = traj(hist_rows(d, s), "aptq_norm_rtg")
        a1.plot(e, al, "-o", color=cmap(i / max(1, len(seeds) - 1)), ms=3.5, lw=1.6, label=f"seed {s} (α={al[-1]:.4f})")
    a1.set_xlabel("Training epoch"); a1.set_ylabel("Coefficient α"); a1.set_xlim(1, 20)
    a1.set_title("(a) Scaled-RTG per training seed", fontsize=10); a1.legend(fontsize=8); a1.grid(True, ls=":", lw=0.5, alpha=0.6)
    variants = [("aptq_norm_rtg", "Scaled RTG", "#d62728", "-"), ("aptq_reward_q", "Reward-quantile", "#1f77b4", "--"),
                ("aptq_moving_dataset_q", "Moving dataset-Q", "#2ca02c", "-."), ("aptq_fixed_reference_q", "Fixed reference-Q", "#9467bd", ":")]
    for pid, lab, c, ls in variants:
        allt = [traj(hist_rows(d, s), pid)[1] for s in seeds]
        ep = list(range(1, len(allt[0]) + 1))
        mean = [sum(x[k] for x in allt) / len(allt) for k in range(len(ep))]
        a2.plot(ep, mean, ls, color=c, lw=1.8, marker="o", ms=3, label=lab)
    a2.set_xlabel("Training epoch"); a2.set_ylabel("Coefficient α (five-seed mean)"); a2.set_xlim(1, 20)
    a2.set_title("(b) All target variants (five-seed mean)", fontsize=10); a2.legend(fontsize=8); a2.grid(True, ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figureS1_trajectories.png"), dpi=300); plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True); p.add_argument("--out-dir", required=True)
    a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    fig2(a.results_dir, a.out_dir); fig3(a.results_dir, a.out_dir); figS1(a.results_dir, a.out_dir)
    print("figures written to", a.out_dir)
