"""Plot script for v2 sweeps that overlays paper's reported plateau values
as horizontal reference lines on Figures 1 and 2.

Reproduces Paper A Figure 1 (PG convergence for varying N) and Figure 2
(PG vs REINFORCE) with the improved q_init setup. Adds paper-target lines
read from Figures 1(a)(b)(c) / 2(a)(b)(c) of the published JAIR version.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 25

# Eyeballed from the paper's Figures 1 & 2 for Bai-PG N=256 plateau:
PAPER_PLATEAU_PG = {2: -7.60, 4: -25.30, 8: -126.0}
# Paper's REINFORCE plateau for N=256 (rough):
PAPER_PLATEAU_REINFORCE = {2: -7.75, 4: -26.0, 8: -128.0}

COLOR_BY_N = {4: "tab:red", 16: "tab:orange", 64: "tab:green",
              256: "tab:blue"}
COLOR_BY_ALGO = {"bai_pg": "tab:blue", "reinforce": "tab:red",
                 "ganesh_npg": "tab:green", "ganesh_mlmc": "tab:purple"}
LABEL_BY_ALGO = {"bai_pg": "Bai PG (Alg 1)",
                 "reinforce": "REINFORCE (eq. 38)",
                 "ganesh_npg": "Ganesh NPG",
                 "ganesh_mlmc": "Ganesh MLMC-NPG"}


def load_runs(dirs):
    runs = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                runs.append(json.load(f))
    return runs


def trailing_mean(stacked, window):
    """stacked: (S, K). Return trailing mean across axis 1."""
    K = stacked.shape[1]
    w = min(window, K)
    cs = np.cumsum(stacked, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    idx_lo = np.maximum(0, np.arange(K) - w + 1)
    idx_hi = np.arange(K) + 1
    counts = (idx_hi - idx_lo).astype(np.float64)
    return (cs[:, idx_hi] - cs[:, idx_lo]) / counts[None, :]


def mean_std(runs, smooth=SMOOTH_WINDOW):
    f_arrays = [np.asarray(r["f"]) for r in runs]
    K = min(len(arr) for arr in f_arrays)
    stacked = np.stack([a[:K] for a in f_arrays], axis=0)
    smoothed = trailing_mean(stacked, smooth)
    return np.arange(K), smoothed.mean(axis=0), smoothed.std(axis=0)


def plot_fig1(runs, M, out_dir):
    filtered = [r for r in runs
                if r["config"]["algorithm"] == "bai_pg"
                and r["config"]["M"] == M]
    if not filtered:
        return
    groups = defaultdict(list)
    for r in filtered:
        groups[r["config"]["N"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N in sorted(groups):
        x, mean, std = mean_std(groups[N])
        color = COLOR_BY_N.get(N)
        ax.plot(x, mean, label=f"N = {N}", color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    if M in PAPER_PLATEAU_PG:
        ax.axhline(PAPER_PLATEAU_PG[M], color="black", linestyle=":",
                   linewidth=1, label=f"paper Bai-PG N=256 ({PAPER_PLATEAU_PG[M]:.1f})")
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f(\hat J^{\pi_k})$")
    ax.set_title(f"Bai-PG convergence, M = {M} (v2, q_0>0)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fig1_pg_sweep_M{M}_v2.png", dpi=150)
    plt.close(fig)


def plot_fig2(runs, M, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, algo in zip(axes, ["bai_pg", "reinforce"]):
        groups = defaultdict(list)
        for r in runs:
            c = r["config"]
            if c["algorithm"] == algo and c["M"] == M:
                groups[c["N"]].append(r)
        if not groups:
            continue
        for N in sorted(groups):
            x, mean, std = mean_std(groups[N])
            color = COLOR_BY_N.get(N)
            ax.plot(x, mean, color=color, label=f"N = {N}")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        target = (PAPER_PLATEAU_PG if algo == "bai_pg"
                  else PAPER_PLATEAU_REINFORCE).get(M)
        if target is not None:
            ax.axhline(target, color="black", linestyle=":", linewidth=1,
                       label=f"paper ({target:.1f})")
        ax.set_xlabel("Outer iteration $k$")
        ax.set_title(LABEL_BY_ALGO[algo])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel(r"$f(\hat J^{\pi_k})$")
    fig.suptitle(f"Bai-PG vs REINFORCE, M = {M} (v2, q_0>0)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fig2_pg_vs_reinforce_M{M}_v2.png", dpi=150)
    plt.close(fig)


def plot_fig3_allalgos(runs, M, N, out_dir):
    present = set(r["config"]["algorithm"]
                  for r in runs if r["config"]["M"] == M)
    if not ({"ganesh_npg", "ganesh_mlmc"} & present):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_f = []
    for algo in ["bai_pg", "reinforce", "ganesh_npg", "ganesh_mlmc"]:
        matched = [r for r in runs
                   if r["config"]["algorithm"] == algo
                   and r["config"]["M"] == M and r["config"]["N"] == N]
        if not matched:
            continue
        x, mean, std = mean_std(matched)
        color = COLOR_BY_ALGO[algo]
        ax.plot(x, mean, label=LABEL_BY_ALGO[algo], color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        all_f.append(mean)
    if all_f:
        flat = np.concatenate(all_f)
        lo, hi = np.percentile(flat, [2, 98])
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    if M in PAPER_PLATEAU_PG:
        ax.axhline(PAPER_PLATEAU_PG[M], color="black", linestyle=":",
                   linewidth=1, label=f"paper PG ({PAPER_PLATEAU_PG[M]:.1f})")
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f(\hat J^{\pi_k})$")
    ax.set_title(f"4-algo comparison, M = {M}, N = {N} (v2)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fig3_all_algos_M{M}_N{N}_v2.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dirs", nargs="+", required=True)
    p.add_argument("--out_dir", default="figures/v2")
    p.add_argument("--Ms", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--fig3_N", type=int, default=64)
    p.add_argument("--policy_filter", default=None,
                   help="restrict plotted runs to this policy kind")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.sweep_dirs)
    if args.policy_filter:
        runs = [r for r in runs if r["config"].get("policy") == args.policy_filter]
    print(f"Loaded {len(runs)} runs from {args.sweep_dirs}"
          f"{' (policy=' + args.policy_filter + ')' if args.policy_filter else ''}")

    for M in args.Ms:
        plot_fig1(runs, M, out_dir)
        plot_fig2(runs, M, out_dir)
        plot_fig3_allalgos(runs, M, args.fig3_N, out_dir)
    print(f"Figures -> {out_dir}")


if __name__ == "__main__":
    main()
