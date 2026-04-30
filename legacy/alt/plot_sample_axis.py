"""Generic plotter that renders a sweep dir's runs in v8 dual-panel + multi-N style,
with x-axis selectable as iteration index ('iter') or cumulative trajectory count
('trajectories').

Usage:
    python alt/plot_sample_axis.py --dir results/sweep_dst \\
        --output figures/dst/dst_3way_traj.png --x_axis trajectories

The plotter infers the (algo, N, seed) grouping from filenames matching the
schema {algo}_{tag}_N{N}_seed{seed}.json (DST/Resource/etc) or
{algo}_{policy}_M{M}_N{N}_alpha{a}_seed{seed}.json (queueing-style).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red",
         "ganesh_mlmc": "tab:purple", "ganesh_npg": "tab:green"}
LABEL = {"bai_pg": "Bai-PG",
         "reinforce": r"REINFORCE ($\delta$=1)",
         "ganesh_mlmc": "MLMC-NPG",
         "ganesh_npg": "Ganesh-NPG"}
ALGO_ORDER = ["bai_pg", "reinforce", "ganesh_mlmc", "ganesh_npg"]


_ALGO_PREFIXES = ("bai_pg", "reinforce", "ganesh_mlmc", "ganesh_npg")


def parse_filename(fn: str):
    """Try to extract (algo, N, seed) from supported filename patterns.
    Supports two schemas:
      Episodic: {algo}_{tag}_N{N}_seed{seed}
      Queueing: {algo}_{policy}_M{M}_N{N}_alpha{a}_seed{seed}
    """
    base = Path(fn).stem
    algo = None
    rest = None
    for a in _ALGO_PREFIXES:
        if base.startswith(a + "_"):
            algo = a
            rest = base[len(a) + 1:]
            break
    if algo is None:
        return None
    m_seed = re.search(r"_seed(\d+)$", rest)
    if not m_seed:
        return None
    seed = int(m_seed.group(1))
    pre = rest[:m_seed.start()]
    m_N = re.search(r"_N(\d+)(?:_alpha[\d.]+)?$", pre)
    if not m_N:
        return None
    N = int(m_N.group(1))
    tag = pre[:m_N.start()]
    return algo, N, seed, tag


def load_runs(sweep_dir: Path) -> Dict[Tuple[str, int], List[dict]]:
    """Returns dict[(algo, N)] = [run dict, ...] across seeds."""
    runs = defaultdict(list)
    for f in sorted(sweep_dir.glob("*.json")):
        if f.name == "sweep_manifest.json":
            continue
        parsed = parse_filename(f.name)
        if parsed is None:
            continue
        algo, N, seed, _ = parsed
        with open(f) as fh:
            d = json.load(fh)
        runs[(algo, N)].append(d)
    return runs


def smooth_mean_std(curves: List[np.ndarray], min_len: int = None):
    if min_len is None:
        min_len = min(len(c) for c in curves)
    arr = np.stack([c[:min_len] for c in curves], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_dual(runs, out_path, x_axis: str = "iter", title: str = "",
              clip_quantile: float = 0.05):
    """1-row panel per N (so for N in {2, 4, 16} we get a 1x3), with mean ± std
    across seeds for each algo. If any curve has values outside [clip_quantile,
    1-clip_quantile] of the global y-range, also produce a dual-panel version
    above with full y-range, below with clipped y-range.
    """
    Ns = sorted({N for (_, N) in runs.keys()})
    if not Ns:
        print(f"No runs found in {out_path.parent}")
        return

    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 4),
                              sharex=False, sharey=False)
    if len(Ns) == 1:
        axes = [axes]

    all_y_vals = []
    for j, N in enumerate(Ns):
        ax = axes[j]
        for algo in ALGO_ORDER:
            seeds = runs.get((algo, N))
            if not seeds:
                continue
            f_curves = [np.asarray(d["f"]) for d in seeds]
            x_curves = ([np.arange(len(c)) for c in f_curves] if x_axis == "iter"
                        else [np.asarray(d["trajectories"]) for d in seeds])
            min_len = min(len(c) for c in f_curves)
            mean, std = smooth_mean_std(f_curves, min_len)
            x_mean = np.stack([c[:min_len] for c in x_curves], axis=0).mean(axis=0)
            ax.plot(x_mean, mean, color=COLOR[algo], label=LABEL[algo], lw=1.5)
            ax.fill_between(x_mean, mean - std, mean + std, color=COLOR[algo], alpha=0.2)
            all_y_vals.extend(mean.tolist())
        ax.set_title(f"N = {N}")
        ax.set_xlabel("cumulative trajectories" if x_axis == "trajectories" else "iteration k")
        ax.set_ylabel("f(J)")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_dual_panel(runs, out_path, x_axis: str = "iter", title: str = ""):
    """Dual-panel (full above, clipped below) — useful when one method diverges.
    1x3 grid each (so 2x3 total).
    """
    Ns = sorted({N for (_, N) in runs.keys()})
    if not Ns:
        return

    fig, axes = plt.subplots(2, len(Ns), figsize=(5 * len(Ns), 7),
                              sharex='col', sharey=False)
    if len(Ns) == 1:
        axes = axes.reshape(2, 1)

    # First, build per-(algo,N) curves and percentiles
    for row in range(2):
        for j, N in enumerate(Ns):
            ax = axes[row, j]
            all_finals = []
            for algo in ALGO_ORDER:
                seeds = runs.get((algo, N))
                if not seeds:
                    continue
                f_curves = [np.asarray(d["f"]) for d in seeds]
                x_curves = ([np.arange(len(c)) for c in f_curves] if x_axis == "iter"
                            else [np.asarray(d["trajectories"]) for d in seeds])
                min_len = min(len(c) for c in f_curves)
                mean, std = smooth_mean_std(f_curves, min_len)
                x_mean = np.stack([c[:min_len] for c in x_curves], axis=0).mean(axis=0)
                ax.plot(x_mean, mean, color=COLOR[algo], label=LABEL[algo], lw=1.5)
                ax.fill_between(x_mean, mean - std, mean + std, color=COLOR[algo], alpha=0.2)
                # Use last 25% of curve for clip stats
                tail = mean[-max(1, min_len // 4):]
                all_finals.extend(tail.tolist())
            if row == 0:
                ax.set_title(f"N = {N} (full)")
            else:
                # Clip y to the inter-quartile range of the tail across all algos
                if all_finals:
                    lo = np.percentile(all_finals, 10)
                    hi = np.percentile(all_finals, 90)
                    span = max(hi - lo, 1.0)
                    ax.set_ylim(lo - 0.5 * span, hi + 0.5 * span)
                ax.set_title(f"N = {N} (clipped)")
            ax.set_xlabel("cumulative trajectories" if x_axis == "trajectories" else "iteration k")
            ax.set_ylabel("f(J)")
            ax.grid(True, alpha=0.3)
            if j == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plateau_table(runs, K_frac: float = 0.1) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Per (algo, N), compute the per-seed mean of f over the last K/10 iters,
    then mean ± std across seeds. Returns dict[(algo, N)] = (mean, std)."""
    out = {}
    for key, seeds in runs.items():
        per_seed = []
        for d in seeds:
            f = np.asarray(d["f"])
            tail = max(1, int(len(f) * K_frac))
            per_seed.append(float(f[-tail:].mean()))
        per_seed = np.asarray(per_seed)
        out[key] = (float(per_seed.mean()), float(per_seed.std()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--x_axis", default="iter", choices=["iter", "trajectories"])
    p.add_argument("--dual_panel", action="store_true")
    p.add_argument("--title", default="")
    p.add_argument("--print_table", action="store_true")
    args = p.parse_args()

    runs = load_runs(Path(args.dir))
    if args.print_table:
        tab = plateau_table(runs)
        print("\nPlateau table (mean +- std, last 10% of K, across seeds):")
        Ns = sorted({N for (_, N) in tab.keys()})
        algos = sorted({a for (a, _) in tab.keys()})
        header = "  N  | " + " | ".join(f"{LABEL.get(a, a):<28}" for a in algos)
        print(header)
        print("-" * len(header))
        for N in Ns:
            row = f"  {N:<3}|"
            for a in algos:
                v = tab.get((a, N))
                if v is None:
                    row += f" {'—':<28}|"
                else:
                    row += f" {v[0]:>10.2f} +- {v[1]:<10.2f}      |"
            print(row)

    if args.dual_panel:
        plot_dual_panel(runs, Path(args.output), x_axis=args.x_axis, title=args.title)
    else:
        plot_dual(runs, Path(args.output), x_axis=args.x_axis, title=args.title)


if __name__ == "__main__":
    main()
