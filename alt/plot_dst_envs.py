"""Visualize the three DST environments side-by-side:
  - Standard 10x5 (Paper B) with 5 treasures, T_max=50
  - Larger 20x10 (project extension) with 10 treasures, T_max=100
  - Original Vamplew 2011 10x11 with 10 treasures, T_max=200 (spec cap 1000)

Outputs:
  figures/dst/dst_envs.png         — side-by-side environment layout
  figures/dst/dst_envs_paths.png   — same with optimal-path overlays
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.envs.dst.small import DST_TREASURES, DST_GRID_ROWS, DST_GRID_COLS, DST_T_MAX
from src.envs.dst.large import (
    LARGE_DST_TREASURES, GRID_ROWS as LARGE_ROWS,
    GRID_COLS as LARGE_COLS, T_MAX as LARGE_T_MAX,
)
from src.envs.dst.original import (
    ORIGINAL_DST_TREASURES, GRID_ROWS as ORIG_ROWS,
    GRID_COLS as ORIG_COLS, T_MAX as ORIG_T_MAX,
)


SEA_COLOR = "#cfe7f5"
DEEP_COLOR = "#7fb6d6"
SEA_FLOOR_COLOR = "#b08968"   # sandy / stone for impassable terrain
SEA_FLOOR_EDGE = "#7d5a3a"
TREASURE_COLOR = "#f5c542"
START_COLOR = "#4caf50"
PATH_COLOR = "#d62728"


def sea_floor_mask(treasures, rows, cols):
    """Vamplew-style: cells below each column's treasure are impassable.

    Returns (rows, cols) bool array; True = sea floor (blocked).
    Assumes every column has at least one treasure.
    """
    depth = {}
    for (r, c) in treasures:
        depth[c] = max(depth.get(c, -1), r)
    mask = np.zeros((rows, cols), dtype=bool)
    for c in range(cols):
        if c in depth:
            for r in range(depth[c] + 1, rows):
                mask[r, c] = True
    return mask


def draw_grid(ax, rows, cols, treasures, title, t_max,
              show_path=False, with_sea_floor=False):
    floor = sea_floor_mask(treasures, rows, cols) if with_sea_floor \
            else np.zeros((rows, cols), dtype=bool)

    # Background gradient: slightly darker as you go deeper (row index up).
    for r in range(rows):
        depth_frac = r / max(1, rows - 1)
        sea_color = tuple(
            (1 - depth_frac) * np.array(_hex2rgb(SEA_COLOR))
            + depth_frac * np.array(_hex2rgb(DEEP_COLOR))
        )
        for c in range(cols):
            if floor[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, rows - 1 - r), 1, 1,
                    facecolor=SEA_FLOOR_COLOR,
                    edgecolor=SEA_FLOOR_EDGE, linewidth=0.5,
                    hatch="//",
                ))
            else:
                ax.add_patch(mpatches.Rectangle(
                    (c, rows - 1 - r), 1, 1,
                    facecolor=sea_color, edgecolor="white", linewidth=0.5,
                ))

    # Treasures.
    for (rr, cc), v in treasures.items():
        ax.add_patch(mpatches.Rectangle(
            (cc, rows - 1 - rr), 1, 1,
            facecolor=TREASURE_COLOR, edgecolor="black", linewidth=1.0,
        ))
        # Value label centered.
        fontsize = 9 if rows <= 10 else 7
        ax.text(cc + 0.5, rows - 1 - rr + 0.5, f"{int(v)}",
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="black")

    # Start cell (0, 0).
    ax.add_patch(mpatches.Rectangle(
        (0, rows - 1), 1, 1,
        facecolor=START_COLOR, edgecolor="black", linewidth=1.0,
        alpha=0.85,
    ))
    fontsize = 8 if rows <= 10 else 6
    ax.text(0.5, rows - 1 + 0.5, "S",
            ha="center", va="center", fontsize=fontsize + 1,
            fontweight="bold", color="white")

    # Optimal path to deepest treasure.
    # Sea-floor-safe route: right along row 0, then dive down deepest column.
    # (The deepest treasure's column has no sea floor below it by construction.)
    if show_path:
        deepest = max(treasures.keys(), key=lambda k: k[0])
        dr, dc = deepest
        if with_sea_floor:
            # Right along row 0 to column dc, then down to (dr, dc).
            path_cells = [(0, c) for c in range(dc + 1)] \
                         + [(r, dc) for r in range(1, dr + 1)]
        else:
            # Open water: down column 0 to row dr, then right.
            path_cells = [(r, 0) for r in range(dr + 1)] \
                         + [(dr, c) for c in range(1, dc + 1)]
        path_x = [c + 0.5 for (_, c) in path_cells]
        path_y = [rows - 1 - r + 0.5 for (r, _) in path_cells]
        ax.plot(path_x, path_y, color=PATH_COLOR, linewidth=2.0,
                linestyle="--", marker="o", markersize=3, alpha=0.9,
                label=f"optimal: R={int(treasures[deepest])}, T={dr+dc}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)

    # Subtitle showing key params.
    ax.set_title(
        f"{title}\n{rows}x{cols} grid, {len(treasures)} treasures, "
        f"T_max={t_max}, |S|={rows*cols}+1",
        fontsize=10,
    )


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def plot_envs(out_path: Path, show_path: bool = False,
              with_sea_floor: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.4, 1.5]})

    draw_grid(axes[0], DST_GRID_ROWS, DST_GRID_COLS, DST_TREASURES,
              "Standard DST (Paper B)", DST_T_MAX,
              show_path=show_path, with_sea_floor=with_sea_floor)
    draw_grid(axes[1], LARGE_ROWS, LARGE_COLS, LARGE_DST_TREASURES,
              "Larger DST (project extension)", LARGE_T_MAX,
              show_path=show_path, with_sea_floor=with_sea_floor)
    draw_grid(axes[2], ORIG_ROWS, ORIG_COLS, ORIGINAL_DST_TREASURES,
              "Original DST (Vamplew 2011)", ORIG_T_MAX,
              show_path=show_path, with_sea_floor=with_sea_floor)

    # Legend patches.
    handles = [
        mpatches.Patch(facecolor=SEA_COLOR, edgecolor="white", label="Sea (depth ~ row)"),
        mpatches.Patch(facecolor=TREASURE_COLOR, edgecolor="black", label="Treasure (value)"),
        mpatches.Patch(facecolor=START_COLOR, edgecolor="black", label="Start (S)"),
    ]
    if with_sea_floor:
        handles.append(mpatches.Patch(
            facecolor=SEA_FLOOR_COLOR, edgecolor=SEA_FLOOR_EDGE,
            hatch="//", label="Sea floor (impassable)",
        ))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    title = "Deep Sea Treasure environments"
    if with_sea_floor:
        title += " (Vamplew-style sea floor)"
    fig.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = Path("figures/dst")
    plot_envs(out_dir / "dst_envs.png",
              show_path=False, with_sea_floor=False)
    plot_envs(out_dir / "dst_envs_paths.png",
              show_path=True, with_sea_floor=False)
    plot_envs(out_dir / "dst_envs_floor.png",
              show_path=False, with_sea_floor=True)
    plot_envs(out_dir / "dst_envs_floor_paths.png",
              show_path=True, with_sea_floor=True)


if __name__ == "__main__":
    main()
