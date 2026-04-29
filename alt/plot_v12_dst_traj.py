"""Plot v12 large-DST final-policy trajectories saved by run_v12_dst_traj.py."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from alt.envs_dst_large import GRID_COLS, GRID_ROWS, LARGE_DST_TREASURES
from alt.run_v12_dst_traj import SEA_FLOOR


RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc)_dst_large_"
    r"(?P<variant>open|floor)_tab_N(?P<N>\d+)_K(?P<K>\d+)_"
    r"(?P<mode>ref|moreK|largerN)_seed(?P<seed>\d+)"
)

LABEL = {
    "ganesh_mlmc": "MLMC-NPG",
    "bai_pg": "Bai-PG",
    "reinforce": "REINFORCE",
}
COLOR = {
    "ganesh_mlmc": "tab:purple",
    "bai_pg": "tab:blue",
    "reinforce": "tab:red",
}
STYLE = {"ref": "-", "moreK": "--", "largerN": ":"}
MARKER = {0: "o", 1: "s", 2: "^"}


def load_runs(path: Path) -> list[dict]:
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
        traj = data["greedy_trajectory"]
        f_vals = np.asarray(data["f"], dtype=float)
        tail = max(1, len(f_vals) // 10)
        runs.append({
            **meta,
            "file": str(fp),
            "trajectory": traj,
            "plateau": float(f_vals[-tail:].mean()),
            "final_f": float(f_vals[-1]),
        })
    return runs


def draw_env(ax, with_floor: bool, title: str) -> None:
    sea_top = np.array([0.81, 0.91, 0.96])
    sea_bottom = np.array([0.50, 0.71, 0.84])
    for r in range(GRID_ROWS):
        frac = r / max(1, GRID_ROWS - 1)
        water = (1 - frac) * sea_top + frac * sea_bottom
        for c in range(GRID_COLS):
            y = GRID_ROWS - 1 - r
            if with_floor and SEA_FLOOR[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, y), 1, 1,
                    facecolor="#b08968", edgecolor="#7d5a3a",
                    linewidth=0.45, hatch="//",
                ))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (c, y), 1, 1, facecolor=water,
                    edgecolor="white", linewidth=0.45,
                ))

    for (rr, cc), value in LARGE_DST_TREASURES.items():
        ax.add_patch(mpatches.Rectangle(
            (cc, GRID_ROWS - 1 - rr), 1, 1,
            facecolor="#f5c542", edgecolor="black", linewidth=1.0, zorder=4,
        ))
        ax.text(cc + 0.5, GRID_ROWS - 1 - rr + 0.5, str(int(value)),
                ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)

    ax.add_patch(mpatches.Rectangle(
        (0, GRID_ROWS - 1), 1, 1,
        facecolor="#4caf50", edgecolor="black", linewidth=1.0, zorder=6,
    ))
    ax.text(0.5, GRID_ROWS - 0.5, "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=7)
    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(GRID_COLS + 1))
    ax.set_yticks(np.arange(GRID_ROWS + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10)


def plot_variant(runs: list[dict], variant: str, out: Path) -> None:
    variant_runs = [r for r in runs if r["variant"] == variant]
    grouped = defaultdict(list)
    for run in variant_runs:
        grouped[run["algo"]].append(run)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10.2))
    algo_order = ["ganesh_mlmc", "bai_pg", "reinforce"]
    with_floor = variant == "floor"

    for ax, algo in zip(axes, algo_order):
        draw_env(ax, with_floor, LABEL[algo])
        for run in sorted(grouped.get(algo, []), key=lambda r: (r["mode"], r["seed"])):
            cells = run["trajectory"]["cells"]
            xs = [c + 0.5 for _, c in cells]
            ys = [GRID_ROWS - 1 - r + 0.5 for r, _ in cells]
            seed = run["seed"]
            mode = run["mode"]
            treasure = run["trajectory"]["treasure"]
            unique_cells = {tuple(cell) for cell in cells}
            status = f"T={int(treasure)}" if run["trajectory"]["terminated"] else "no T"
            label = (
                f"{mode} seed{seed}: {status}, "
                f"{run['trajectory']['steps']} steps, f~{run['plateau']:.2f}"
            )
            if len(unique_cells) == 1:
                # A stationary policy otherwise renders as a tiny marker hidden
                # by the start cell. Draw a visible ring and annotate the action.
                ax.scatter(xs[0], ys[0], s=120 + 35 * seed,
                           facecolors="none", edgecolors=COLOR[algo],
                           linewidths=2.0, marker=MARKER.get(seed, "o"),
                           label=label, zorder=9)
                first_action = run["trajectory"]["action_names"][0]
                ax.text(xs[0] + 0.18, ys[0] - 0.20 - 0.22 * seed,
                        f"seed{seed}: {first_action} x{run['trajectory']['steps']}",
                        color=COLOR[algo], fontsize=7, zorder=10)
            else:
                ax.plot(xs, ys, color=COLOR[algo], linestyle=STYLE[mode],
                        marker=MARKER.get(seed, "o"), markersize=3,
                        linewidth=1.8, alpha=0.85, label=label, zorder=8)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16),
                  fontsize=7, frameon=False)

    handles = [
        mpatches.Patch(facecolor="#cfe7f5", edgecolor="white", label="open water"),
        mpatches.Patch(facecolor="#f5c542", edgecolor="black", label="terminal treasure"),
    ]
    if with_floor:
        handles.append(mpatches.Patch(
            facecolor="#b08968", edgecolor="#7d5a3a", hatch="//",
            label="seafloor/blocked",
        ))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    subtitle = "with Vamplew-style seafloor" if with_floor else "without seafloor"
    fig.suptitle(
        f"v12 Large DST final-policy greedy trajectories, gamma=0.9999, {subtitle}\n"
        "Paths are replayed from saved final policies for every seed/config",
        fontsize=12, y=1.07,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def write_summary(runs: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("variant,algo,mode,N,K,seed,plateau,final_f,terminated,treasure,steps,file\n")
        for run in sorted(runs, key=lambda r: (
            r["variant"], r["algo"], r["mode"], r["seed"], r["N"], r["K"],
        )):
            traj = run["trajectory"]
            f.write(
                f"{run['variant']},{run['algo']},{run['mode']},{run['N']},{run['K']},"
                f"{run['seed']},{run['plateau']},{run['final_f']},"
                f"{traj['terminated']},{traj['treasure']},{traj['steps']},{run['file']}\n"
            )
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="results/sweep_v12_dst_traj_gamma09999")
    p.add_argument("--out_dir", default="figures/v12_dst_traj_gamma09999")
    p.add_argument("--summary", default="results/v12_summaries/dst_traj_gamma09999.csv")
    args = p.parse_args()

    runs = load_runs(Path(args.sweep_dir))
    write_summary(runs, Path(args.summary))
    plot_variant(runs, "open", Path(args.out_dir) / "dst_large_open_learned_trajectories.png")
    plot_variant(runs, "floor", Path(args.out_dir) / "dst_large_floor_learned_trajectories.png")


if __name__ == "__main__":
    main()
