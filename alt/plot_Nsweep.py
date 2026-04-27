"""Plot N-sweep results: one figure per policy class with two panels —
  (a) convergence curves at N=4 (or chosen N), three algorithms overlaid
  (b) plateau f-value as a function of N, three algorithms overlaid
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 25

COLOR = {"bai_pg": "tab:blue",
         "reinforce": "tab:red",
         "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai PG",
         "reinforce": r"REINFORCE ($\delta$=1)",
         "ganesh_mlmc": "MLMC-NPG"}


def load(dirs):
    runs = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                runs.append(json.load(f))
    return runs


def trailing_mean(stack, w=SMOOTH_WINDOW):
    K = stack.shape[1]; w = min(w, K)
    cs = np.cumsum(stack, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    lo = np.maximum(0, np.arange(K) - w + 1); hi = np.arange(K) + 1
    counts = (hi - lo).astype(np.float64)
    return (cs[:, hi] - cs[:, lo]) / counts[None, :]


def plateau_stats(runs):
    out = defaultdict(list)
    for r in runs:
        c = r["config"]
        f = np.asarray(r["f"])
        last = f[-max(1, f.shape[0]//10):].mean()
        out[(c["algorithm"], c["N"])].append(last)
    summary = {}
    for (algo, N), vals in out.items():
        vals = np.array(vals)
        summary[(algo, N)] = (vals.mean(), vals.std(), len(vals))
    return summary


def plot_policy(runs, title, out_path, focus_N=4, ylim=None):
    # Left: convergence at focus_N
    # Right: plateau vs N
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    # Left: convergence at focus_N.
    focus = [r for r in runs if r["config"]["N"] == focus_N]
    grouped = defaultdict(list)
    for r in focus:
        grouped[r["config"]["algorithm"]].append(r)
    for algo, rs in grouped.items():
        fs = [np.asarray(r["f"]) for r in rs]
        K = min(len(f) for f in fs)
        stack = np.stack([f[:K] for f in fs], axis=0)
        sm = trailing_mean(stack)
        m, s = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(K)
        ax1.plot(x, m, color=COLOR[algo], label=LABEL[algo], lw=1.8)
        ax1.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
    ax1.set_xlabel("Outer iteration $k$")
    ax1.set_ylabel(r"$f_{\alpha=2}(\hat J^{\pi_k})$")
    ax1.set_title(f"Convergence at N={focus_N}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", frameon=False)

    # Right: plateau vs N.
    plats = plateau_stats(runs)
    algos = ["bai_pg", "reinforce", "ganesh_mlmc"]
    Ns = sorted({N for (_, N) in plats})
    for algo in algos:
        xs = []
        means = []
        stds = []
        for N in Ns:
            if (algo, N) in plats:
                m, s, _ = plats[(algo, N)]
                xs.append(N); means.append(m); stds.append(s)
        xs = np.array(xs); means = np.array(means); stds = np.array(stds)
        ax2.errorbar(xs, means, yerr=stds, fmt="o-",
                     color=COLOR[algo], label=LABEL[algo],
                     lw=1.8, markersize=7, capsize=4)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch size $N$")
    ax2.set_ylabel(r"Plateau $f_{\alpha=2}$ (last 10% of $K$)")
    ax2.set_title("Plateau vs. batch size")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=False)
    if ylim:
        ax1.set_ylim(*ylim); ax2.set_ylim(*ylim)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # Print table
    print(f"\n== {title} ==")
    print(f"{'N':>6} | "
          + " | ".join(f"{LABEL[a]:<20}" for a in algos))
    print("-" * 90)
    for N in Ns:
        row = f"{N:>6} | "
        for algo in algos:
            if (algo, N) in plats:
                m, s, n = plats[(algo, N)]
                row += f"{m:>8.2f} ± {s:>5.2f}({n})   "
            else:
                row += f"{'—':>18}   "
        print(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--focus_N", type=int, default=4)
    args = p.parse_args()

    runs = load([args.sweep_dir])
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    plot_policy(runs, args.title, args.out_path, focus_N=args.focus_N)
