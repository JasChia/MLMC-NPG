"""Plot v12 open-water small/original DST trajectories and summaries."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from alt.plot_v11_fair_budget import load_runs as load_training_runs
from alt.plot_v11_fair_budget import plot_case, summarise, write_summary_csv
from alt.run_v12_dst_open_variants import SEA_FLOORS, SPECS


RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc)_"
    r"(?P<env>dst_open|dst_floor|dst_original_open)_tab_"
    r"N(?P<N>\d+)_K(?P<K>\d+)_(?P<mode>ref|moreK|largerN)_seed(?P<seed>\d+)"
)

LABEL = {"ganesh_mlmc": "MLMC-NPG", "bai_pg": "Bai-PG", "reinforce": "REINFORCE"}
COLOR = {"ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue", "reinforce": "tab:red"}
STYLE = {"ref": "-", "moreK": "--", "largerN": ":"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def load_trajectory_runs(path: Path) -> list[dict]:
    runs = []
    for fp in sorted(path.glob("*.json")):
        if fp.name == "sweep_manifest.json":
            continue
        match = RUN_RE.match(fp.stem)
        if not match:
            print(f"skip unmatched {fp.name}")
            continue
        with open(fp) as f:
            data = json.load(f)
        meta = match.groupdict()
        meta["N"] = int(meta["N"])
        meta["K"] = int(meta["K"])
        meta["seed"] = int(meta["seed"])
        f_vals = np.asarray(data["f"], dtype=float)
        tail = max(1, len(f_vals) // 10)
        runs.append({
            **meta,
            "file": str(fp),
            "trajectory": data["greedy_trajectory"],
            "plateau": float(f_vals[-tail:].mean()),
            "final_f": float(f_vals[-1]),
        })
    return runs


def draw_env(ax, env: str, title: str) -> None:
    spec = SPECS[env]
    sea_top = np.array([0.81, 0.91, 0.96])
    sea_bottom = np.array([0.50, 0.71, 0.84])
    for r in range(spec.rows):
        frac = r / max(1, spec.rows - 1)
        water = (1 - frac) * sea_top + frac * sea_bottom
        for c in range(spec.cols):
            if env.endswith("_floor") and SEA_FLOORS[env][r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, spec.rows - 1 - r), 1, 1,
                    facecolor="#b08968", edgecolor="#7d5a3a",
                    linewidth=0.5, hatch="//",
                ))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (c, spec.rows - 1 - r), 1, 1,
                    facecolor=water, edgecolor="white", linewidth=0.5,
                ))
    for (rr, cc), value in spec.treasures.items():
        ax.add_patch(mpatches.Rectangle(
            (cc, spec.rows - 1 - rr), 1, 1,
            facecolor="#f5c542", edgecolor="black", linewidth=1.0, zorder=4,
        ))
        ax.text(cc + 0.5, spec.rows - 1 - rr + 0.5, str(int(value)),
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)
    ax.add_patch(mpatches.Rectangle(
        (0, spec.rows - 1), 1, 1,
        facecolor="#4caf50", edgecolor="black", linewidth=1.0, zorder=6,
    ))
    ax.text(0.5, spec.rows - 0.5, "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=7)
    ax.set_xlim(0, spec.cols)
    ax.set_ylim(0, spec.rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(spec.cols + 1))
    ax.set_yticks(np.arange(spec.rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10)


def plot_env_trajectories(runs: list[dict], env: str, out: Path) -> None:
    env_runs = [r for r in runs if r["env"] == env]
    grouped = defaultdict(list)
    for run in env_runs:
        grouped[run["algo"]].append(run)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8.2))
    for ax, algo in zip(axes, ["ganesh_mlmc", "bai_pg", "reinforce"]):
        draw_env(ax, env, LABEL[algo])
        for idx, run in enumerate(sorted(grouped.get(algo, []), key=lambda r: (r["mode"], r["seed"]))):
            cells = run["trajectory"]["cells"]
            xs = [c + 0.5 for _, c in cells]
            ys = [SPECS[env].rows - 1 - r + 0.5 for r, _ in cells]
            unique_cells = {tuple(cell) for cell in cells}
            seed = run["seed"]
            mode = run["mode"]
            marker = MARKERS[idx % len(MARKERS)]
            treasure = run["trajectory"]["treasure"]
            status = f"T={int(treasure)}" if run["trajectory"]["terminated"] else "no T"
            label = (
                f"{mode} seed{seed}: {status}, "
                f"{run['trajectory']['steps']} steps, f~{run['plateau']:.2f}"
            )
            if len(unique_cells) == 1:
                ax.scatter(xs[0], ys[0], s=120 + 12 * idx, facecolors="none",
                           edgecolors=COLOR[algo], linewidths=1.8,
                           marker=marker, label=label, zorder=9)
                first_action = run["trajectory"]["action_names"][0]
                ax.text(xs[0] + 0.18, ys[0] - 0.18 - 0.14 * idx,
                        f"s{seed}: {first_action}x{run['trajectory']['steps']}",
                        color=COLOR[algo], fontsize=6, zorder=10)
            else:
                ax.plot(xs, ys, color=COLOR[algo], linestyle=STYLE[mode],
                        marker=marker, markersize=3, linewidth=1.7,
                        alpha=0.82, label=label, zorder=8)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20),
                  fontsize=6, frameon=False)

    handles = [
        mpatches.Patch(facecolor="#cfe7f5", edgecolor="white", label="open water"),
        mpatches.Patch(facecolor="#f5c542", edgecolor="black", label="terminal treasure"),
    ]
    if env.endswith("_floor"):
        handles.append(mpatches.Patch(
            facecolor="#b08968", edgecolor="#7d5a3a", hatch="//",
            label="seafloor/blocked",
        ))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    if env == "dst_open":
        title = "Small DST open-water"
    elif env == "dst_floor":
        title = "Small DST with seafloor"
    else:
        title = "Original DST open-water"
    fig.suptitle(
        f"v12 {title} final-policy greedy trajectories, gamma=0.9999\n"
        "Paths are replayed from saved final policies for seeds 3-12",
        fontsize=12, y=1.07,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def write_traj_summary(runs: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "algo", "mode", "N", "K", "seed", "plateau",
                         "final_f", "terminated", "treasure", "steps", "file"])
        for run in sorted(runs, key=lambda r: (r["env"], r["algo"], r["mode"], r["seed"])):
            traj = run["trajectory"]
            writer.writerow([
                run["env"], run["algo"], run["mode"], run["N"], run["K"],
                run["seed"], run["plateau"], run["final_f"],
                traj["terminated"], traj["treasure"], traj["steps"], run["file"],
            ])
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="results/sweep_v12_dst_open_variants_gamma09999_seed3_12")
    p.add_argument("--out_dir", default="figures/v12_dst_open_variants_gamma09999_seed3_12")
    p.add_argument("--summary", default="results/v12_summaries/dst_open_variants_gamma09999_seed3_12.csv")
    p.add_argument("--plateaus", default="results/v12_summaries/dst_open_variants_gamma09999_seed3_12_plateaus.csv")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    traj_runs = load_trajectory_runs(sweep_dir)
    write_traj_summary(traj_runs, Path(args.summary))
    for env in ["dst_open", "dst_floor", "dst_original_open"]:
        plot_env_trajectories(traj_runs, env, Path(args.out_dir) / f"{env}_learned_trajectories.png")

    training_runs = load_training_runs(sweep_dir)
    rows = summarise(training_runs)
    write_summary_csv(rows, Path(args.plateaus))
    titles = {
        "dst_open": "v12 Small DST open-water gamma=0.9999 tabular fair-budget (seeds 3-12)",
        "dst_floor": "v12 Small DST with seafloor gamma=0.9999 tabular fair-budget (seeds 3-12)",
        "dst_original_open": "v12 Original DST open-water gamma=0.9999 tabular fair-budget (seeds 3-12)",
    }
    for env, title in titles.items():
        plot_case([r for r in training_runs if r["env"] == env and r["policy"] == "tab"],
                  title, Path(args.out_dir) / f"{env}_fair_budget.png")


if __name__ == "__main__":
    main()
