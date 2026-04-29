"""v14 plots — fair-budget comparison for one configuration / tag.

Reads results/sweep_v14_*/  and produces:
  figures/v14/<tag>_fair_budget.png   curves vs cumulative trajectories
  figures/v14/<tag>_plateaus.csv      mean/std plateau per algo
  figures/v14/<tag>_pvalues.txt       MLMC vs each alt rank-sum

Filename convention (from run_v14_search.py):
  {algo}_{env_tag}_{policy_tag}_B{B}_N{N}_K{K_use}_beta{beta}_lr{lr}_v14_{tag}_seed{s}.json
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
from scipy import stats


RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc|scalarized_reinforce)_"
    r"(?P<env_tag>.+?)_"
    r"(?P<policy>tab|nn\d+(?:x\d+)*)_"
    r"B(?P<B>\d+)_N(?P<N>\d+)_K(?P<K>\d+)_"
    r"beta(?P<beta>[^_]+)_lr(?P<lr>[^_]+)(?:_o[^_]+)?_"
    r"v(?P<version>1[45])_(?P<tag>.+)_seed(?P<seed>\d+)"
)

LABEL = {
    "ganesh_mlmc": "MLMC-NPG", "bai_pg": "Bai-PG",
    "reinforce": "REINFORCE (paper A)",
    "scalarized_reinforce": "REINFORCE (scalarized)",
}
COLOR = {
    "ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
    "reinforce": "tab:red", "scalarized_reinforce": "tab:orange",
}
ALGO_ORDER = ["ganesh_mlmc", "bai_pg", "reinforce", "scalarized_reinforce"]


def load_runs(path: Path, tag: str | None = None) -> list[dict]:
    runs = []
    for fp in sorted(path.glob("*.json")):
        if fp.name.startswith("sweep_manifest"):
            continue
        m = RUN_RE.match(fp.stem)
        if not m:
            continue
        if tag and m.group("tag") != tag:
            continue
        with open(fp) as f:
            d = json.load(f)
        meta = m.groupdict()
        meta["seed"] = int(meta["seed"])
        f_arr = np.asarray(d["f"], dtype=float)
        traj = np.asarray(d["trajectories"], dtype=int)
        tail = max(1, len(f_arr) // 10)
        runs.append({
            **meta,
            "f": f_arr, "trajectories": traj,
            "final_f": float(f_arr[-1]),
            "plateau": float(f_arr[-tail:].mean()),
        })
    return runs


def aggregate(runs: list[dict]) -> dict:
    by_algo = defaultdict(list)
    for r in runs:
        by_algo[r["algo"]].append(r)
    out = {}
    for algo, rs in by_algo.items():
        if not rs:
            continue
        t_lo = max(r["trajectories"][0] for r in rs)
        t_hi = min(r["trajectories"][-1] for r in rs)
        grid = np.linspace(t_lo, t_hi, 200)
        curves = np.stack([np.interp(grid, r["trajectories"], r["f"]) for r in rs])
        out[algo] = {
            "traj_grid": grid,
            "mean": curves.mean(axis=0),
            "std": curves.std(axis=0, ddof=1) if len(rs) > 1 else np.zeros_like(grid),
            "n_seeds": len(rs),
        }
    return out


def plot_curves(curves: dict, runs: list[dict], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGO_ORDER:
        if algo not in curves:
            continue
        c = curves[algo]
        ax.plot(c["traj_grid"], c["mean"], color=COLOR[algo],
                label=f"{LABEL[algo]} (n={c['n_seeds']})", linewidth=1.7)
        ax.fill_between(c["traj_grid"], c["mean"] - c["std"],
                        c["mean"] + c["std"], color=COLOR[algo], alpha=0.18)
    ax.set_xlabel("trajectories")
    ax.set_ylabel("f(J)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_plateau_csv(runs: list[dict], out_path: Path, tag: str) -> None:
    by_algo = defaultdict(list)
    for r in runs:
        by_algo[r["algo"]].append(r["plateau"])
    rows = []
    for algo in ALGO_ORDER:
        vals = by_algo.get(algo, [])
        if not vals:
            continue
        rows.append({
            "tag": tag, "algo": algo, "n_seeds": len(vals),
            "plateau_mean": float(np.mean(vals)),
            "plateau_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "algo", "n_seeds",
                                            "plateau_mean", "plateau_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


def write_pvalues(runs: list[dict], out_path: Path, tag: str) -> None:
    by_algo = defaultdict(list)
    for r in runs:
        by_algo[r["algo"]].append(r["final_f"])
    if "ganesh_mlmc" not in by_algo:
        print(f"[skip pvalues] no MLMC runs for {tag}")
        return
    mlmc = np.asarray(by_algo["ganesh_mlmc"], dtype=np.float64)
    lines = [f"{tag}: MLMC-NPG vs alternatives (final f, MWU exact two-sided)"]
    lines.append(f"  MLMC: n={len(mlmc)} mean={mlmc.mean():.3f} sd={mlmc.std(ddof=1) if len(mlmc) > 1 else 0:.3f}")
    for algo in ALGO_ORDER:
        if algo == "ganesh_mlmc" or algo not in by_algo:
            continue
        vals = np.asarray(by_algo[algo], dtype=np.float64)
        try:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided", method="exact")
        except ValueError:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided", method="asymptotic")
        diff = float(mlmc.mean() - vals.mean())
        lines.append(
            f"  vs {algo:<22} n={len(vals)} mean={vals.mean():7.3f} "
            f"sd={vals.std(ddof=1) if len(vals) > 1 else 0:6.3f}  "
            f"M-X={diff:+7.3f}  p={res.pvalue:.4f}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", default="figures/v14")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    runs = load_runs(Path(args.results_dir), tag=args.tag)
    if not runs:
        print(f"no runs found for tag={args.tag} in {args.results_dir}")
        return
    print(f"loaded {len(runs)} runs for tag={args.tag}")
    title = args.title or args.tag
    curves = aggregate(runs)
    out_dir = Path(args.out_dir)
    plot_curves(curves, runs, out_dir / f"{args.tag}_fair_budget.png", title)
    write_plateau_csv(runs, out_dir / f"{args.tag}_plateaus.csv", args.tag)
    write_pvalues(runs, out_dir / f"{args.tag}_pvalues.txt", args.tag)


if __name__ == "__main__":
    main()
