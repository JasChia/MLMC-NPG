"""v13 Fruit Tree alpha-fairness plots.

Outputs (under figures/v13/):
  fruit_tree_alphas_fair_budget.png   per-alpha curves (one panel per alpha)
  fruit_tree_alphas_plateau_vs_alpha.png  plateau vs alpha for each method
  fruit_tree_alphas_plateaus.csv      tabular summary

Note on the alpha=0 case: U_alpha is linear (utilitarian sum), so the
problem is concave but not strictly. MLMC's variance reduction has no
leverage there. The plot lets you see this empirically.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc|scalarized_reinforce)_fruit_tree_"
    r"a(?P<alpha_tag>\d+)_tab_"
    r"N(?P<N>\d+)_K(?P<K>\d+)_(?P<mode>v13)_seed(?P<seed>\d+)"
)

LABEL = {
    "ganesh_mlmc": "MLMC-NPG", "bai_pg": "Bai-PG",
    "reinforce": "REINFORCE (paper A eq.38)",
    "scalarized_reinforce": "REINFORCE (scalarized)",
}
COLOR = {
    "ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
    "reinforce": "tab:red", "scalarized_reinforce": "tab:orange",
}
ALGO_ORDER = ["ganesh_mlmc", "bai_pg", "reinforce", "scalarized_reinforce"]


def alpha_from_tag(tag: str) -> float:
    return int(tag) / 100.0


def load_runs(path: Path) -> list[dict]:
    runs = []
    for fp in sorted(path.glob("*.json")):
        if fp.name == "sweep_manifest.json":
            continue
        m = RUN_RE.match(fp.stem)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        meta = m.groupdict()
        meta["alpha"] = alpha_from_tag(meta["alpha_tag"])
        meta["N"] = int(meta["N"])
        meta["K"] = int(meta["K"])
        meta["seed"] = int(meta["seed"])
        f_vals = np.asarray(d["f"], dtype=float)
        traj = np.asarray(d["trajectories"], dtype=int)
        tail = max(1, len(f_vals) // 10)
        runs.append({
            **meta,
            "f": f_vals, "trajectories": traj,
            "final_f": float(f_vals[-1]),
            "plateau": float(f_vals[-tail:].mean()),
        })
    return runs


def aggregate_curves(runs: list[dict]) -> dict:
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["algo"], r["alpha"])].append(r)
    out = {}
    for key, rs in by_key.items():
        t_lo = max(r["trajectories"][0] for r in rs)
        t_hi = min(r["trajectories"][-1] for r in rs)
        grid = np.linspace(t_lo, t_hi, 100)
        curves = np.stack([np.interp(grid, r["trajectories"], r["f"]) for r in rs])
        out[key] = {
            "traj_grid": grid,
            "mean": curves.mean(axis=0),
            "std": curves.std(axis=0, ddof=1) if len(rs) > 1 else np.zeros_like(grid),
            "n_seeds": len(rs),
        }
    return out


def plot_per_alpha_panels(runs: list[dict], curves: dict, out_path: Path) -> None:
    alphas = sorted({r["alpha"] for r in runs})
    n = len(alphas)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, alpha in zip(axes, alphas):
        for algo in ALGO_ORDER:
            key = (algo, alpha)
            if key not in curves:
                continue
            c = curves[key]
            ax.plot(c["traj_grid"], c["mean"], color=COLOR[algo],
                    label=f"{LABEL[algo]} (n={c['n_seeds']})", linewidth=1.6)
            ax.fill_between(c["traj_grid"], c["mean"] - c["std"],
                            c["mean"] + c["std"], color=COLOR[algo], alpha=0.15)
        ax.set_title(f"alpha = {alpha}", fontsize=11)
        ax.set_xlabel("trajectories", fontsize=9)
        ax.set_ylabel("f(J)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(alphas):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle("v13 Fruit Tree (depth=7) — alpha-fairness sweep, fair budget ~5k traj",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_plateau_vs_alpha(runs: list[dict], out_path: Path) -> None:
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["algo"], r["alpha"])].append(r["plateau"])
    alphas = sorted({a for (_, a) in by_key.keys()})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ALGO_ORDER:
        means = []
        stds = []
        for a in alphas:
            vals = by_key.get((algo, a), [])
            if not vals:
                means.append(np.nan); stds.append(np.nan); continue
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(alphas, means, yerr=stds, color=COLOR[algo], marker="o",
                    capsize=3, label=LABEL[algo], linewidth=1.5)
    ax.set_xlabel("alpha (alpha-fairness exponent)")
    ax.set_ylabel("plateau f(J)  (mean +/- sd, last K/10)")
    ax.set_title("Fruit Tree plateau vs alpha (v13 fair budget)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_plateau_csv(runs: list[dict], out_path: Path) -> None:
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["algo"], r["alpha"])].append(r["plateau"])
    rows = []
    for (algo, alpha), vals in sorted(by_key.items()):
        rows.append({
            "alpha": alpha, "algo": algo, "n_seeds": len(vals),
            "plateau_mean": float(np.mean(vals)),
            "plateau_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "algo", "n_seeds",
                                            "plateau_mean", "plateau_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/sweep_v13_fruit_tree_alphas")
    p.add_argument("--out_dir", default="figures/v13")
    args = p.parse_args()

    runs = load_runs(Path(args.results_dir))
    if not runs:
        print(f"no runs found in {args.results_dir}")
        return
    print(f"loaded {len(runs)} runs from {args.results_dir}")
    curves = aggregate_curves(runs)
    out_dir = Path(args.out_dir)
    plot_per_alpha_panels(runs, curves, out_dir / "fruit_tree_alphas_fair_budget.png")
    plot_plateau_vs_alpha(runs, out_dir / "fruit_tree_alphas_plateau_vs_alpha.png")
    write_plateau_csv(runs, out_dir / "fruit_tree_alphas_plateaus.csv")


if __name__ == "__main__":
    main()
