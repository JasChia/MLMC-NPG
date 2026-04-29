"""v13 DST plots — 6-panel comparison across (small/large/original) x (open/floor).

Outputs (under figures/v13/):
  dst_<policy_tag>_fair_budget.png      6-panel: f vs trajectories, mean+/-std
  dst_<policy_tag>_plateaus.csv         tabular plateau summary
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
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc|scalarized_reinforce)_"
    r"(?P<env>dst_open|dst_floor|dst_large_open|dst_large_floor|dst_original_open|dst_original_floor)_"
    r"(?P<policy>tab|nn\d+(?:x\d+)*)_"
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

ENV_TITLES = {
    "dst_open":           "10x5 DST  open",
    "dst_floor":          "10x5 DST  floor",
    "dst_large_open":     "20x10 DST  open",
    "dst_large_floor":    "20x10 DST  floor",
    "dst_original_open":  "10x11 DST  open",
    "dst_original_floor": "10x11 DST  floor",
}
ENV_ORDER = list(ENV_TITLES.keys())


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
    """key = (algo, env) -> {'traj_grid': arr, 'mean': arr, 'std': arr, 'n_seeds': int}.

    Re-samples each run to a common trajectory grid so seeds with different
    random Bk path (MLMC) are alignable.
    """
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["algo"], r["env"])].append(r)
    out = {}
    for key, rs in by_key.items():
        if not rs:
            continue
        # Common trajectory grid: from max(min) to min(max), N=100
        t_lo = max(r["trajectories"][0] for r in rs)
        t_hi = min(r["trajectories"][-1] for r in rs)
        grid = np.linspace(t_lo, t_hi, 100)
        curves = []
        for r in rs:
            curves.append(np.interp(grid, r["trajectories"], r["f"]))
        curves = np.stack(curves)
        out[key] = {
            "traj_grid": grid,
            "mean": curves.mean(axis=0),
            "std": curves.std(axis=0, ddof=1) if len(rs) > 1 else np.zeros_like(grid),
            "n_seeds": len(rs),
        }
    return out


def plot_six_panel(curves: dict, out_path: Path, policy_tag: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=False)
    axes = axes.flatten()
    for ax, env in zip(axes, ENV_ORDER):
        for algo in ALGO_ORDER:
            key = (algo, env)
            if key not in curves:
                continue
            c = curves[key]
            ax.plot(c["traj_grid"], c["mean"], color=COLOR[algo],
                    label=f"{LABEL[algo]} (n={c['n_seeds']})", linewidth=1.6)
            ax.fill_between(c["traj_grid"], c["mean"] - c["std"],
                            c["mean"] + c["std"], color=COLOR[algo], alpha=0.15)
        ax.set_title(ENV_TITLES[env], fontsize=11)
        ax.set_xlabel("trajectories", fontsize=9)
        ax.set_ylabel("f(J)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)
    # one combined legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle(f"v13 DST — fair budget ~5k traj, policy={policy_tag}, "
                 f"gamma=0.9999  (MLMC: B_max=16, B_inner=4, N_inner=35, K=35;  "
                 f"Bai: N1=N2=4, K=612;  REINFORCE: N=4, K=1225)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_plateau_csv(runs: list[dict], out_path: Path, policy_tag: str) -> None:
    """One row per (algo, env): plateau mean +/- std across seeds."""
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["algo"], r["env"])].append(r["plateau"])
    rows = []
    for env in ENV_ORDER:
        for algo in ALGO_ORDER:
            vals = by_key.get((algo, env), [])
            if not vals:
                continue
            rows.append({
                "env": env, "algo": algo, "policy": policy_tag,
                "n_seeds": len(vals),
                "plateau_mean": float(np.mean(vals)),
                "plateau_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "algo", "policy",
                                            "n_seeds", "plateau_mean", "plateau_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default="figures/v13")
    p.add_argument("--policy_tag", default="tab",
                   help="label for plot title and filename (e.g. 'tab' or 'nn32')")
    args = p.parse_args()

    runs = load_runs(Path(args.results_dir))
    if not runs:
        print(f"no runs found in {args.results_dir}")
        return
    print(f"loaded {len(runs)} runs from {args.results_dir}")
    curves = aggregate_curves(runs)
    out_dir = Path(args.out_dir)
    plot_six_panel(curves, out_dir / f"dst_{args.policy_tag}_fair_budget.png",
                    args.policy_tag)
    write_plateau_csv(runs, out_dir / f"dst_{args.policy_tag}_plateaus.csv",
                       args.policy_tag)


if __name__ == "__main__":
    main()
