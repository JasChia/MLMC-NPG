"""v13 DST greedy-trajectory plots — one PNG per env, 3 panels (one per algo).

Each panel draws the env grid (with sea floor where applicable), treasures,
and overlays the final-policy greedy trajectory for each seed.

Outputs (under figures/v13/):
  trajectories/{env}_{policy_tag}_learned.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from alt.run_v13_dst import SPECS, SEA_FLOORS


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
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


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
        if "greedy_trajectory" not in d:
            continue
        meta = m.groupdict()
        meta["seed"] = int(meta["seed"])
        meta["N"] = int(meta["N"])
        meta["K"] = int(meta["K"])
        f_arr = np.asarray(d["f"], dtype=float)
        tail = max(1, len(f_arr) // 10)
        runs.append({
            **meta,
            "trajectory": d["greedy_trajectory"],
            "plateau": float(f_arr[-tail:].mean()),
            "final_f": float(f_arr[-1]),
        })
    return runs


def draw_grid(ax, env: str) -> None:
    spec = SPECS[env]
    floor = SEA_FLOORS[env] if spec.has_floor else None
    sea_top = np.array([0.81, 0.91, 0.96])
    sea_bot = np.array([0.50, 0.71, 0.84])
    for r in range(spec.rows):
        frac = r / max(1, spec.rows - 1)
        water = (1 - frac) * sea_top + frac * sea_bot
        for c in range(spec.cols):
            if floor is not None and floor[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, spec.rows - 1 - r), 1, 1,
                    facecolor="#b08968", edgecolor="#7d5a3a",
                    linewidth=0.5, hatch="//"))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (c, spec.rows - 1 - r), 1, 1,
                    facecolor=water, edgecolor="white", linewidth=0.5))
    for (rr, cc), v in spec.treasures.items():
        ax.add_patch(mpatches.Rectangle(
            (cc, spec.rows - 1 - rr), 1, 1,
            facecolor="#f5c542", edgecolor="black", linewidth=1.0, zorder=4))
        ax.text(cc + 0.5, spec.rows - 1 - rr + 0.5, str(int(v)),
                ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)
    ax.add_patch(mpatches.Rectangle(
        (0, spec.rows - 1), 1, 1,
        facecolor="#4caf50", edgecolor="black", linewidth=1.0, zorder=6))
    ax.text(0.5, spec.rows - 0.5, "S", ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=7)
    ax.set_xlim(0, spec.cols)
    ax.set_ylim(0, spec.rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_env(runs: list[dict], env: str, out: Path, policy_tag: str) -> None:
    spec = SPECS[env]
    fig_w = max(15, 3 * 5)
    fig_h = 1 + spec.rows * 0.45
    fig, axes = plt.subplots(1, 4, figsize=(4 * 5, max(5, fig_h)))
    grouped = defaultdict(list)
    for r in runs:
        if r["env"] == env and r["policy"] == policy_tag:
            grouped[r["algo"]].append(r)
    for ax, algo in zip(axes, ALGO_ORDER):
        draw_grid(ax, env)
        ax.set_title(f"{LABEL[algo]}", fontsize=11)
        for idx, run in enumerate(sorted(grouped.get(algo, []), key=lambda r: r["seed"])):
            cells = run["trajectory"]["cells"]
            xs = [c + 0.5 for _, c in cells]
            ys = [spec.rows - 1 - r + 0.5 for r, _ in cells]
            uniq = {tuple(c) for c in cells}
            seed = run["seed"]
            marker = MARKERS[idx % len(MARKERS)]
            treasure = run["trajectory"]["treasure"]
            status = f"T={int(treasure)}" if run["trajectory"]["terminated"] else "no T"
            label = f"s{seed}: {status}, {run['trajectory']['steps']} steps, f~{run['plateau']:.1f}"
            if len(uniq) == 1:
                ax.scatter(xs[0], ys[0], s=80 + 16 * idx,
                           facecolors="none", edgecolors=COLOR[algo],
                           linewidths=1.8, marker=marker, zorder=9, label=label)
            else:
                ax.plot(xs, ys, color=COLOR[algo],
                        marker=marker, markersize=3, linewidth=1.7,
                        alpha=0.85, label=label, zorder=8)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7, frameon=False)

    handles = [
        mpatches.Patch(facecolor="#cfe7f5", edgecolor="white", label="water"),
        mpatches.Patch(facecolor="#f5c542", edgecolor="black", label="terminal treasure"),
    ]
    if spec.has_floor:
        handles.append(mpatches.Patch(facecolor="#b08968", edgecolor="#7d5a3a",
                                        hatch="//", label="seafloor (blocked)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"v13 {env} ({policy_tag}) final-policy greedy trajectories, gamma=0.9999",
                 fontsize=11, y=1.07)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default="figures/v13/trajectories")
    p.add_argument("--policy_tag", default="tab")
    args = p.parse_args()

    runs = load_runs(Path(args.results_dir))
    if not runs:
        print(f"no runs found in {args.results_dir}")
        return
    print(f"loaded {len(runs)} runs from {args.results_dir}")
    for env in SPECS.keys():
        envs_in_dir = {r["env"] for r in runs}
        if env not in envs_in_dir:
            continue
        plot_env(runs, env, Path(args.out_dir) / f"{env}_{args.policy_tag}_learned.png",
                 args.policy_tag)


if __name__ == "__main__":
    main()
