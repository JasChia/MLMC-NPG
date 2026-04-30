"""Produce Paper A Figures 1-2 style plots + a three-way algorithm comparison.

Reads JSON results from a sweep directory (see run_sweep.py) and emits:
    figures/fig1_pg_sweep_M{M}.png
    figures/fig2_pg_vs_reinforce_M{M}.png
    figures/fig3_all_algos_M{M}_N{N}.png     (if Ganesh runs are present)

Usage:
    python plot_results.py --sweep_dir results/sweep --out_dir figures
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 25  # moving-average window applied per run before mean/std

COLOR_BY_N = {4: "tab:red", 16: "tab:orange", 64: "tab:green", 256: "tab:blue"}
COLOR_BY_POLICY = {"log_linear": "tab:blue", "tabular": "tab:orange", "nn": "tab:green"}
LABEL_BY_POLICY = {"log_linear": "Log-linear softmax",
                   "tabular": "Tabular softmax",
                   "nn": "NN (MLP 64×64)"}
COLOR_BY_ALGO = {
    "bai_pg": "tab:blue",
    "reinforce": "tab:red",
    "ganesh_npg": "tab:green",
    "ganesh_mlmc": "tab:purple",
}
LABEL_BY_ALGO = {
    "bai_pg": "Bai PG (Alg 1)",
    "reinforce": "REINFORCE (eq. 38)",
    "ganesh_npg": "Ganesh NPG",
    "ganesh_mlmc": "Ganesh MLMC-NPG",
}


def load_runs(sweep_dir: Path):
    """Return list of history dicts (each embeds the run config)."""
    runs = []
    for p in sorted(sweep_dir.glob("*.json")):
        if p.name == "sweep_manifest.json":
            continue
        with open(p) as f:
            h = json.load(f)
        runs.append(h)
    return runs


def load_runs_from_dirs(sweep_dirs: list[Path]):
    """Load runs from multiple directories (for cross-policy comparison plots)."""
    runs = []
    for d in sweep_dirs:
        runs.extend(load_runs(Path(d)))
    return runs


def group_by(runs, keys):
    """Group runs by the tuple of `keys` from run['config']."""
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        g[tuple(c[k] for k in keys)].append(r)
    return g


def _mean_std_vs_iter(runs_list, smooth_window: int = 1):
    """runs_list: list of histories. Return (x=iter_idx, mean_f, std_f).

    If smooth_window > 1, apply a centered moving-average to each run BEFORE
    computing mean/std across runs. This matches Paper A's cleaner plots.
    """
    f_arrays = [np.asarray(r["f"]) for r in runs_list]
    K = min(len(arr) for arr in f_arrays)
    stacked = np.stack([a[:K] for a in f_arrays], axis=0)  # (S, K)
    if smooth_window > 1:
        w = min(smooth_window, K)
        # Trailing running mean: smoothed[t] = mean(raw[max(0, t-w+1) : t+1]).
        cs = np.cumsum(stacked, axis=1)
        cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
        idx_lo = np.maximum(0, np.arange(K) - w + 1)
        idx_hi = np.arange(K) + 1
        counts = (idx_hi - idx_lo).astype(np.float64)
        stacked = (cs[:, idx_hi] - cs[:, idx_lo]) / counts[None, :]
    return np.arange(K), stacked.mean(axis=0), stacked.std(axis=0)


def _mean_std_vs_traj(runs_list, n_grid=200):
    """Interpolate each run onto a shared cumulative-trajectory axis, then mean/std."""
    traj_all = [np.asarray(r["trajectories"]) for r in runs_list]
    f_all = [np.asarray(r["f"]) for r in runs_list]
    # Build a common log-spaced traj axis within the overlap range.
    lo = max(t[0] for t in traj_all)
    hi = min(t[-1] for t in traj_all)
    if hi <= lo:
        return None
    x = np.linspace(lo, hi, n_grid)
    interped = np.stack([np.interp(x, t, f) for t, f in zip(traj_all, f_all)], axis=0)
    return x, interped.mean(axis=0), interped.std(axis=0)


def plot_fig1(runs, M, out_dir: Path):
    """Paper A Figure 1: Bai-PG convergence for varying N, single M."""
    filtered = [r for r in runs if r["config"]["algorithm"] == "bai_pg"
                and r["config"]["M"] == M]
    if not filtered:
        return
    groups = defaultdict(list)
    for r in filtered:
        groups[r["config"]["N"]].append(r)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for N in sorted(groups):
        x, mean, std = _mean_std_vs_iter(groups[N], smooth_window=SMOOTH_WINDOW)
        color = COLOR_BY_N.get(N, None)
        ax.plot(x, mean, label=f"N = {N}", color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f(\hat J^{\pi_k})$")
    ax.set_title(f"Bai-PG convergence, M = {M} (Paper A Figure 1 reproduction)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig1_pg_sweep_M{M}.png", dpi=150)
    plt.close(fig)


def plot_fig2(runs, M, out_dir: Path):
    """Paper A Figure 2: Bai-PG vs REINFORCE for varying N.

    Two-panel layout: Bai-PG on the left, REINFORCE on the right, shared y-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, algo in zip(axes, ["bai_pg", "reinforce"]):
        groups = defaultdict(list)
        for r in runs:
            c = r["config"]
            if c["algorithm"] == algo and c["M"] == M:
                groups[c["N"]].append(r)
        for N in sorted(groups):
            x, mean, std = _mean_std_vs_iter(groups[N], smooth_window=SMOOTH_WINDOW)
            color = COLOR_BY_N.get(N, None)
            ax.plot(x, mean, color=color, label=f"N = {N}")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("Outer iteration $k$")
        ax.set_title(LABEL_BY_ALGO[algo])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel(r"$f(\hat J^{\pi_k})$")
    fig.suptitle(f"Bai-PG vs REINFORCE, M = {M} (Paper A Figure 2 reproduction)")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig2_pg_vs_reinforce_M{M}.png", dpi=150)
    plt.close(fig)


def plot_fig3_all_algos(runs, M, N, out_dir: Path):
    """Extension: all four algorithms compared at fixed (M, N).

    Uses the same trailing moving-average smoothing as Fig 1/2. Y-limits are
    clamped to the 5-95 percentile band of the raw data to keep MLMC's
    transient spikes (single-trajectory estimate when Q_bar=0) from
    stretching the axis and burying the mean curves.
    """
    present = set(r["config"]["algorithm"] for r in runs if r["config"]["M"] == M)
    if not ({"ganesh_npg", "ganesh_mlmc"} & present):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    all_f_values = []
    for ax, xaxis_kind in zip(axes, ["iter", "trajectories"]):
        for algo in ["bai_pg", "reinforce", "ganesh_npg", "ganesh_mlmc"]:
            matched = [r for r in runs
                       if r["config"]["algorithm"] == algo
                       and r["config"]["M"] == M and r["config"]["N"] == N]
            if not matched:
                continue
            color = COLOR_BY_ALGO[algo]
            label = LABEL_BY_ALGO[algo]
            if xaxis_kind == "iter":
                x, mean, std = _mean_std_vs_iter(matched, smooth_window=SMOOTH_WINDOW)
            else:
                res = _mean_std_vs_traj(matched)
                if res is None:
                    continue
                x, mean, std = res
            all_f_values.append(mean)
            ax.plot(x, mean, label=label, color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.grid(True, alpha=0.3)
        if xaxis_kind == "iter":
            ax.set_xlabel("Outer iteration $k$")
            ax.set_title(f"vs. iteration, M={M}, N={N}")
            ax.set_ylabel(r"$f(\hat J^{\pi_k})$")
        else:
            ax.set_xlabel("Cumulative trajectories")
            ax.set_xscale("log")
            ax.set_title(f"vs. trajectory count, M={M}, N={N}")
    # Clamp y-limits to the 2-98 percentile band of smoothed means.
    if all_f_values:
        flat = np.concatenate(all_f_values)
        lo, hi = np.percentile(flat, [2, 98])
        pad = 0.1 * (hi - lo + 1e-9)
        for ax in axes:
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Three-way comparison at M = {M}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig3_all_algos_M{M}_N{N}.png", dpi=150)
    plt.close(fig)


def plot_fig4_policy_comparison(runs, M, N, algorithm, out_dir: Path):
    """Figure 4: same algorithm, different policy classes, at fixed (M, N).

    Overlays log_linear / tabular / nn runs to show how policy-class choice
    interacts with the algorithm.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted_any = False
    all_f_values = []
    for policy in ["log_linear", "tabular", "nn"]:
        matched = [r for r in runs
                   if r["config"]["algorithm"] == algorithm
                   and r["config"]["policy"] == policy
                   and r["config"]["M"] == M and r["config"]["N"] == N]
        if not matched:
            continue
        plotted_any = True
        d_val = matched[0]["config"].get("d", None)
        # Not always present in config; infer from the policy if absent.
        label = LABEL_BY_POLICY.get(policy, policy)
        if d_val is not None:
            label = f"{label} (d={d_val})"
        x, mean, std = _mean_std_vs_iter(matched, smooth_window=SMOOTH_WINDOW)
        color = COLOR_BY_POLICY.get(policy)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        all_f_values.append(mean)
    if not plotted_any:
        plt.close(fig)
        return
    if all_f_values:
        flat = np.concatenate(all_f_values)
        lo, hi = np.percentile(flat, [2, 98])
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f(\hat J^{\pi_k})$")
    ax.set_title(f"Policy-class comparison: {LABEL_BY_ALGO[algorithm]}, M={M}, N={N}")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig4_policy_M{M}_N{N}_{algorithm}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default=None,
                   help="single sweep directory (legacy; use --sweep_dirs for multi)")
    p.add_argument("--sweep_dirs", nargs="+", default=None,
                   help="multiple sweep directories to load runs from")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--Ms", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--fig3_N", type=int, default=64)
    p.add_argument("--fig4_algos", nargs="+", default=["bai_pg"],
                   help="algorithms to overlay for the policy-comparison plot")
    p.add_argument("--fig4_Ns", type=int, nargs="+", default=[64])
    p.add_argument("--policy_filter", default=None,
                   help="restrict plotted runs to a single policy (log_linear/tabular/nn)")
    args = p.parse_args()

    assert args.sweep_dir or args.sweep_dirs, "provide --sweep_dir or --sweep_dirs"
    dirs = [args.sweep_dir] if args.sweep_dir else args.sweep_dirs
    runs = load_runs_from_dirs([Path(d) for d in dirs])
    if args.policy_filter:
        runs = [r for r in runs if r["config"].get("policy") == args.policy_filter]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(runs)} runs from {dirs}"
          f"{' (filter=' + args.policy_filter + ')' if args.policy_filter else ''}")

    for M in args.Ms:
        plot_fig1(runs, M, out_dir)
        plot_fig2(runs, M, out_dir)
        plot_fig3_all_algos(runs, M, args.fig3_N, out_dir)
        for N in args.fig4_Ns:
            for algo in args.fig4_algos:
                plot_fig4_policy_comparison(runs, M, N, algo, out_dir)
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
