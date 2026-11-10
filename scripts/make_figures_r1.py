# -*- coding: utf-8 -*-
"""IJIES-format figures for the first revision: single panels, no panel labels, text >= 10 pt at final printed size,
no outer borders. Fig. 1 workflow, Fig. 2 trade-off, Fig. 3 coefficient trajectories, Fig. 4 validation landscape.
Usage: python make_figures_r1.py --results-dir <archive>/results_clean_200k --out-dir .
"""
import os, csv, argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "axes.linewidth": 0.8})

def load_metrics(d):
    return {r["policy_id"]: r for r in csv.DictReader(open(os.path.join(d, "summary_metrics.csv")))}

def fig1(out):
    # one-column workflow (8.2 cm wide at print size), vertical snake layout
    bw, bh, gx, gy = 1.38, 0.56, 0.30, 0.28
    W, H = 2 * bw + gx, 4 * bh + 3 * gy
    fig = plt.figure(figsize=(W + 0.08, H + 0.08)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.04, W + 0.04); ax.set_ylim(-0.04, H + 0.04); ax.axis("off")
    xl, xr = 0.0, bw + gx
    ys = [H - bh - i * (bh + gy) for i in range(4)]
    def box(x, y, t):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="black", lw=1.0))
        ax.text(x + bw / 2, y + bh / 2, t, ha="center", va="center", fontsize=10, linespacing=1.05)
    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.0, color="black"))
    box(xl, ys[0], "EdNet-KT4\nlearner logs");              box(xr, ys[0], "Tabular MDP\n(s, a, r, s\u2032)")
    box(xl, ys[1], "Scaled RTG\ntarget q$_{0.90}$");         box(xr, ys[1], "Offline Q, CQL,\nAPTQ-inspired")
    box(xl, ys[2], "FQE / OPE on\nuntouched test set");     box(xr, ys[2], "Bellman MSE,\nsupport, match")
    box(xl, ys[3], "Scoped empirical\ninterpretation")
    m = bh / 2
    arrow((xl + bw + 0.03, ys[0] + m), (xr - 0.03, ys[0] + m))
    arrow((xr + bw / 2, ys[0] - 0.03), (xr + bw / 2, ys[1] + bh + 0.03))
    arrow((xl + bw + 0.03, ys[1] + m), (xr - 0.03, ys[1] + m))
    arrow((xr + bw / 2, ys[1] - 0.03), (xr + bw / 2, ys[2] + bh + 0.03))
    arrow((xr - 0.03, ys[2] + m), (xl + bw + 0.03, ys[2] + m))
    arrow((xl + bw / 2, ys[2] - 0.03), (xl + bw / 2, ys[3] + bh + 0.03))
    fig.savefig(os.path.join(out, "figure1_workflow.png"), dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def fig2(d, out):
    m = load_metrics(d)
    def g(pid, base):
        r = m[pid]; mean = float(r[base + "_mean"]); lo = float(r[base + "_ci95_lo"]); hi = float(r[base + "_ci95_hi"])
        return mean, (hi - lo) / 2
    order = [("offline_q", "Offline Q", "#7f7f7f", "o"), ("cql_020", "CQL α = 0.20", "#1f77b4", "s"),
             ("cql_val_mse", "CQL tuned", "#2ca02c", "^"), ("cql_final_alpha_matched", "CQL α-matched", "#9467bd", "D"),
             ("aptq_norm_rtg", "APTQ scaled RTG", "#d62728", "*")]
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    for pid, name, c, mk in order:
        x, xe = g(pid, "action_match"); y, ye = g(pid, "fqe_value_visitation")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, color=c, ms=11 if mk == "*" else 6, capsize=2, elinewidth=0.8,
                    mec="k", mew=0.5, label=name, zorder=3)
    ax.set_xlabel("Action match"); ax.set_ylabel("FQE value")
    ax.set_xlim(0.1, 1.0); ax.set_ylim(1.5, 11.5)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.4); fig.savefig(os.path.join(out, "figure2_tradeoff.png"), dpi=300); plt.close(fig)

def hist_rows(d, seed):
    return list(csv.DictReader(open(os.path.join(d, f"seed_{seed}_histories.csv"))))

def traj(rows, pid):
    r = sorted([x for x in rows if x["policy_id"] == pid], key=lambda x: int(x["epoch"]))
    return [int(x["epoch"]) for x in r], [float(x["alpha"]) for x in r]

def fig3(d, out):
    h = hist_rows(d, 2026)
    series = [("aptq_norm_rtg", "Scaled RTG", "#d62728", "-"), ("aptq_reward_q", "Reward quantile", "#1f77b4", "--"),
              ("aptq_moving_dataset_q", "Moving dataset Q", "#2ca02c", "-."), ("aptq_fixed_reference_q", "Fixed reference Q", "#9467bd", ":")]
    seeds = [2026, 2027, 2028, 2029, 2030]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for pid, lab, c, ls in series:
        allt = [traj(hist_rows(d, s), pid)[1] for s in seeds]
        ep = list(range(1, len(allt[0]) + 1)); mean = [sum(x[k] for x in allt) / len(allt) for k in range(len(ep))]
        ax.plot(ep, mean, ls, color=c, lw=1.6, marker="o", ms=3, label=lab)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Coefficient α"); ax.set_xlim(1, 20); ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper left", frameon=False, labelspacing=0.25)
    fig.tight_layout(pad=0.4); fig.savefig(os.path.join(out, "figure3_trajectories.png"), dpi=300); plt.close(fig)

def fig4(d, out):
    grid = [r for r in csv.DictReader(open(os.path.join(d, "selection_grid.csv"))) if r["family"] == "aptq_norm_rtg" and r["scale"] in ("0.05", "0.1", "0.2")]
    by = defaultdict(list)
    for r in grid: by[float(r["scale"])].append((float(r["quantile"]), float(r["bellman_mse_val"])))
    cols = {0.05: "#1f77b4", 0.1: "#d62728", 0.2: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for sc in sorted(by):
        pts = sorted(by[sc]); ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=cols[sc], lw=1.5, ms=4, label=f"Scale {sc:g}")
    best = min((p[1] for p in by[0.1]))
    ax.scatter([0.9], [best], s=150, marker="*", color="gold", ec="k", lw=0.8, zorder=5, label="Selected")
    ax.set_xlabel("Target quantile"); ax.set_ylabel("Validation Bellman MSE"); ax.set_xticks([0.6, 0.8, 0.9])
    ax.set_ylim(1.82, 2.62)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper center", ncol=2, frameon=False, labelspacing=0.25, columnspacing=1.2)
    fig.tight_layout(pad=0.4); fig.savefig(os.path.join(out, "figure4_landscape.png"), dpi=300); plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--results-dir", required=True); p.add_argument("--out-dir", required=True)
    a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    fig1(a.out_dir); fig2(a.results_dir, a.out_dir); fig3(a.results_dir, a.out_dir); fig4(a.results_dir, a.out_dir)
    print("done")
