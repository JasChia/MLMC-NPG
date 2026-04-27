"""Plotter for the MLMC-showcase sweep: per (alpha, M, N) panel with
iteration-axis AND cumulative-trajectory-axis views. The trajectory-axis
view is the fair comparison — it controls for per-outer-iter compute
differences between the algorithms (MLMC uses more trajectories per
iteration than Bai-PG, so equal-K comparisons favour Bai-PG).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 25

COLOR_BY_ALGO = {"bai_pg": "tab:blue", "reinforce": "tab:red",
                 "ganesh_npg": "tab:green", "ganesh_mlmc": "tab:purple"}
LABEL_BY_ALGO = {"bai_pg": "Bai PG (biased plug-in)",
                 "reinforce": "REINFORCE (eq. 38)",
                 "ganesh_npg": "Ganesh NPG (unbiased + 2nd-order smooth)",
                 "ganesh_mlmc": "Ganesh MLMC-NPG (unbiased, no 2nd-order)"}


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
    K = stacked.shape[1]
    w = min(window, K)
    cs = np.cumsum(stacked, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    idx_lo = np.maximum(0, np.arange(K) - w + 1)
    idx_hi = np.arange(K) + 1
    counts = (idx_hi - idx_lo).astype(np.float64)
    return (cs[:, idx_hi] - cs[:, idx_lo]) / counts[None, :]


def mean_std_vs_iter(runs):
    arrs = [np.asarray(r["f"]) for r in runs]
    K = min(a.shape[0] for a in arrs)
    stack = np.stack([a[:K] for a in arrs], axis=0)
    stack = trailing_mean(stack, SMOOTH_WINDOW)
    return np.arange(K), stack.mean(axis=0), stack.std(axis=0)


def mean_std_vs_traj(runs, n_grid=250):
    traj_all = [np.asarray(r["trajectories"]) for r in runs]
    f_all = [np.asarray(r["f"]) for r in runs]
    lo = max(t[0] for t in traj_all)
    hi = min(t[-1] for t in traj_all)
    if hi <= lo:
        return None
    x = np.linspace(lo, hi, n_grid)
    interped = np.stack([np.interp(x, t, f) for t, f in zip(traj_all, f_all)], axis=0)
    smoothed = trailing_mean(interped, SMOOTH_WINDOW)
    return x, smoothed.mean(axis=0), smoothed.std(axis=0)


def plot_one(runs, alpha, M, N, out_dir):
    filtered = [r for r in runs
                if r["config"]["alpha"] == alpha
                and r["config"]["M"] == M and r["config"]["N"] == N]
    if not filtered:
        return
    grouped = defaultdict(list)
    for r in filtered:
        grouped[r["config"]["algorithm"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: vs outer iteration
    ax = axes[0]
    for algo, rs in grouped.items():
        x, mean, std = mean_std_vs_iter(rs)
        c = COLOR_BY_ALGO[algo]
        ax.plot(x, mean, color=c, label=LABEL_BY_ALGO[algo])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f_\alpha(\hat J^{\pi_k})$")
    ax.set_title(f"vs. outer iteration (favours short-per-iter methods)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    # Right: vs cumulative trajectories (fair)
    ax = axes[1]
    for algo, rs in grouped.items():
        res = mean_std_vs_traj(rs)
        if res is None:
            continue
        x, mean, std = res
        c = COLOR_BY_ALGO[algo]
        ax.plot(x, mean, color=c, label=LABEL_BY_ALGO[algo])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("Cumulative trajectories")
    ax.set_xscale("log")
    ax.set_title("vs. cumulative trajectories (fair per-sample comparison)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(f"MLMC showcase: M = {M}, N = {N}, "
                 fr"$\alpha$ = {alpha}")
    fig.tight_layout()
    fig.savefig(out_dir / f"mlmc_M{M}_N{N}_alpha{alpha:.1f}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dirs", nargs="+", required=True)
    p.add_argument("--out_dir", default="figures/v3_mlmc")
    args = p.parse_args()

    runs = load_runs(args.sweep_dirs)
    print(f"Loaded {len(runs)} runs")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all (alpha, M, N) tuples present:
    cfgs = sorted({(r["config"]["alpha"], r["config"]["M"], r["config"]["N"])
                   for r in runs})
    for alpha, M, N in cfgs:
        plot_one(runs, alpha, M, N, out_dir)
    print(f"Figures -> {out_dir}")


if __name__ == "__main__":
    main()
