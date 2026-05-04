"""Generate breakable_bottles_b_sweep.tikz directly (no tikzplotlib dependency).

Produces a 1×2 pgfplots groupplot (panels for B=4, B=8) plus a standalone axis
(B=16) positioned below, with a shared legend underneath the whole figure.
Reads JSONs from ../data/breakable_bottles/ and writes the .tikz to ../.

Algorithms plotted: MLMC-NPG, Policy Gradient, REINFORCE.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LABEL = {
    "mlmc_npg": "MLMC-NPG",
    "bai_pg":   "Policy Gradient",
    "reinforce": "REINFORCE",
}
TIKZ_COLOR = {
    "mlmc_npg": "mediumpurple148103189",
    "bai_pg":   "steelblue31119180",
    "reinforce": "crimson2143940",
}
ORDER = list(LABEL.keys())

# (B value, panel letter)
B_PANELS = [(4, "(a)"), (8, "(b)"), (16, "(c)")]

RUN_RE = re.compile(
    r"^(?P<algo>mlmc_npg|bai_pg|reinforce)_breakable_bottles_"
    r"\S*?_B(?P<B>\d+)_\S*_seed(?P<seed>\d+)\.json$"
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "breakable_bottles"
OUT  = HERE.parent / "breakable_bottles_b_sweep.tikz"

N_GRID = 200

# Trajectory-budget cap: each panel is shown out to MLMC-NPG's K_BUDGET_ITER
# cumulative trajectory count (averaged across seeds). This matches the
# original figure, which froze the comparison at K_mlmc=40 even when the raw
# JSONs ran for more outer iterations (e.g. the K_mlmc=200 B=8 sweep).
K_BUDGET_ITER = 40


def load_runs(B):
    """Return {algo: list of {f, traj}} for runs with the given B.

    Prefers data with K_mlmc closest to K_BUDGET_ITER per (algo, seed) so that,
    when both K_mlmc=40 and K_mlmc=200 runs exist for the same B, only one set
    is used. Falls back to whatever's available otherwise.
    """
    by_seed = defaultdict(dict)  # (algo, seed) -> list of (K_proximity, payload)
    for fp in sorted(DATA.iterdir()):
        m = RUN_RE.match(fp.name)
        if not m or int(m.group("B")) != B or m.group("algo") not in ORDER:
            continue
        d = json.load(open(fp))
        K = int(d.get("config", {}).get("K", len(d["f"])))
        payload = {
            "f":    np.asarray(d["f"], float),
            "traj": np.asarray(d["trajectories"], int),
            "K":    K,
        }
        algo = m.group("algo")
        proximity = abs(K - K_BUDGET_ITER) if algo == "mlmc_npg" else 0
        key = (algo, m.group("seed"))
        existing = by_seed[key].get("proximity", float("inf"))
        if proximity < existing or "payload" not in by_seed[key]:
            by_seed[key] = {"proximity": proximity, "payload": payload, "algo": algo}

    runs = defaultdict(list)
    for entry in by_seed.values():
        runs[entry["algo"]].append(entry["payload"])
    return runs


def interp_runs(runs):
    """Returns {algo: (grid, mean, std, n)}.

    Each curve is truncated at MLMC-NPG's K_BUDGET_ITER cumulative-trajectory
    count (averaged across MLMC seeds) so all algorithms share a single
    fair-budget horizontal axis, matching the original figure.
    """
    mlmc_runs = runs.get("mlmc_npg", [])
    if not mlmc_runs:
        return {}
    cap = float(np.mean([
        r["traj"][min(K_BUDGET_ITER - 1, len(r["traj"]) - 1)]
        for r in mlmc_runs
    ]))

    out = {}
    for algo in ORDER:
        rs = runs.get(algo, [])
        if not rs:
            continue
        # Per-seed: keep only the part of the curve within the shared budget.
        clipped = []
        for r in rs:
            mask = r["traj"] <= cap
            if mask.sum() < 2:                 # need at least two points to interp
                continue
            clipped.append((r["traj"][mask], r["f"][mask]))
        if not clipped:
            continue
        t_lo = max(t[0]  for t, _ in clipped)
        t_hi = min(t[-1] for t, _ in clipped)
        if t_lo >= t_hi:
            continue
        grid   = np.linspace(t_lo, t_hi, N_GRID)
        curves = np.stack([np.interp(grid, t, y) for t, y in clipped])
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        out[algo] = (grid, mean, std, len(clipped))
    return out


def band_path(grid, mean, std, color):
    upper = mean + std
    lower = mean - std
    lines = [f"\\path [draw={color}, fill={color}, opacity=0.18]"]
    lines.append(f"(axis cs:{grid[0]:.6g},{upper[0]:.6g})")
    for x, y in zip(grid[1:], upper[1:]):
        lines.append(f"--(axis cs:{x:.6g},{y:.6g})")
    for x, y in zip(grid[::-1], lower[::-1]):
        lines.append(f"--(axis cs:{x:.6g},{y:.6g})")
    lines.append("--cycle;")
    return "\n".join(lines)


def mean_line(grid, mean, color):
    lines = [f"\\addplot [line width=0.68pt, {color}]", "table {%"]
    for x, y in zip(grid, mean):
        lines.append(f"{x:.6g} {y:.6g}")
    lines.append("};")
    return "\n".join(lines)


def shared_yrange(panel_data):
    """Common y-axis range across all panels so they're visually comparable."""
    lo = min((m - s).min() for d in panel_data.values()
             for (_, m, s, _) in d.values())
    hi = max((m + s).max() for d in panel_data.values()
             for (_, m, s, _) in d.values())
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def panel_xrange(data):
    xlo = min(d[0][0]  for d in data.values())
    xhi = max(d[0][-1] for d in data.values())
    pad = (xhi - xlo) * 0.05
    return xlo - pad, xhi + pad


def emit_curves(blocks, data):
    """Append band paths and mean lines for each algo (in ORDER)."""
    for algo in ORDER:
        if algo not in data:
            continue
        grid, mean, std, _ = data[algo]
        blocks.append(band_path(grid, mean, std, TIKZ_COLOR[algo]))
    for algo in ORDER:
        if algo not in data:
            continue
        grid, mean, _, _ = data[algo]
        blocks.append(mean_line(grid, mean, TIKZ_COLOR[algo]))


HEADER = r"""% Generated by plot_breakable_bottles.py
\begin{tikzpicture}

\definecolor{crimson2143940}{RGB}{214,39,40}
\definecolor{darkgray176}{RGB}{176,176,176}
\definecolor{lightgray204}{RGB}{204,204,204}
\definecolor{mediumpurple148103189}{RGB}{148,103,189}
\definecolor{steelblue31119180}{RGB}{31,119,180}

\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.5cm}]
"""

FOOTER = r"""\node[anchor=north] at ([yshift=-0.3cm]current bounding box.south)
  {\pgfplotslegendfromname{sharedlegend}};
\end{tikzpicture}
"""


def main():
    panel_data = {}
    for B, _ in B_PANELS:
        runs = load_runs(B)
        data = interp_runs(runs)
        if not data:
            print(f"WARNING: no data for B={B}")
            continue
        panel_data[B] = data

    if not panel_data:
        raise SystemExit("No data found in {}; run run_experiments.py first.".format(DATA))

    ylo, yhi = shared_yrange(panel_data)

    blocks = [HEADER]

    # --- Panels (a) B=4 and (b) B=8: shared groupplot ---
    for idx, (B, panel) in enumerate(B_PANELS[:2]):
        data = panel_data.get(B)
        if data is None:
            continue
        xlo, xhi = panel_xrange(data)

        opts = ["\\nextgroupplot[", "height=6cm,"]
        if idx == 0:
            opts += [
                "legend cell align={left},",
                "legend style={",
                "  font=\\footnotesize,",
                "  /tikz/every even column/.append style={column sep=5pt},",
                "},",
                "legend to name=sharedlegend,",
                "legend columns=3,",
            ]
        else:
            opts.append("legend style={opacity=0, at={(0,0)}, overlay},")

        opts += [
            "tick align=outside,",
            "tick pos=left,",
            f"title={{\\textbf{{{panel}}} $B = {B}$}},",
            "width=7cm,",
            "xlabel={Sampled Trajectories},",
            f"xmin={xlo:.6g}, xmax={xhi:.6g},",
            "xtick style={color=black},",
            "ylabel={Breakable Bottles: $f(R, t, b)$},",
            f"ymin={ylo:.10g}, ymax={yhi:.10g},",
            "ytick style={color=black}",
            "]",
        ]
        blocks.append("\n".join(opts))
        emit_curves(blocks, data)

        if idx == 0:
            for algo in ORDER:
                if algo in data:
                    blocks.append(f"\\addlegendentry{{{LABEL[algo]}}}")
        blocks.append("")

    blocks.append(r"\end{groupplot}")
    blocks.append("")

    # --- Panel (c) B=16: standalone axis below the groupplot ---
    if 16 in panel_data:
        data = panel_data[16]
        xlo, xhi = panel_xrange(data)

        opts = [
            r"\begin{axis}[",
            r"at={($(group c1r1.south)!0.5!(group c2r1.south)-(0,1.8cm)$)},",
            "anchor=north,",
            "height=6cm,",
            "legend style={opacity=0, at={(0,0)}, overlay},",
            "tick align=outside,",
            "tick pos=left,",
            r"title={\textbf{(c)} $B = 16$},",
            "width=7cm,",
            "xlabel={Sampled Trajectories},",
            f"xmin={xlo:.6g}, xmax={xhi:.6g},",
            "xtick={0,4000,8000,12000},",
            "scaled x ticks=false,",
            "xtick style={color=black},",
            f"ymin={ylo:.10g}, ymax={yhi:.10g},",
            "ylabel={Breakable Bottles: $f(R, t, b)$},",
            "ytick style={color=black}",
            "]",
        ]
        blocks.append("\n".join(opts))
        emit_curves(blocks, data)
        blocks.append(r"\end{axis}")
        blocks.append("")

    blocks.append(FOOTER)

    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {OUT}")

    # ── Companion PNG via matplotlib ──────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_color = {"mlmc_npg": "tab:purple", "bai_pg": "tab:blue", "reinforce": "tab:red"}
    png_out   = HERE.parent / "breakable_bottles_b_sweep.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (B, panel) in zip(axes, B_PANELS):
        data = panel_data.get(B)
        if data is None:
            ax.set_title(f"{panel} B = {B}  (no data)")
            ax.set_xlabel("Sampled Trajectories")
            continue
        for algo in ORDER:
            if algo not in data:
                continue
            grid, mean, std, n = data[algo]
            ax.plot(grid, mean, color=png_color[algo], lw=1.7,
                    label=f"{LABEL[algo]} (n={n})")
            ax.fill_between(grid, mean - std, mean + std,
                            color=png_color[algo], alpha=0.18)
        ax.set_title(rf"{panel}  $B = {B}$")
        ax.set_xlabel("Sampled Trajectories")
        ax.set_ylabel(r"Breakable Bottles: $f(R, t, b)$")
        ax.set_ylim(ylo, yhi)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
