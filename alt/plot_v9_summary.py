"""Cross-environment summary for the v9 envs (Resource Gathering, Multi-DST,
Larger DST, Episodic queueing) plus existing v8 large-d cases.

Renders one symlog bar chart of MLMC's plateau-margin over Bai-PG at small N
across all envs, organized by (env, d).
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

from alt.plot_sample_axis import load_runs, plateau_table


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}


SETTINGS = [
    # (label, sweep_dir, d, target_N)
    ("DST  M=2  d=204",          "results/sweep_dst",                  204, 16),
    ("Resource M=3 d=404",       "results/sweep_v9_resource",          404, 16),
    ("Multi-DST M=6 d=204",      "results/sweep_v9_dst_multi",         204, 4),
    ("Larger-DST M=2 d=804",     "results/sweep_v9_dst_large",         804, 4),
    ("EpQ M=8 d=72 (FAILURE)",   "results/sweep_v9_episodic_queue",    72, 2),
    ("logL M=8 d=72",            "results/sweep_v5_loglinear_Nsweep",  72, 2),
    ("logL M=16 d=272",          "results/sweep_v7_M16_alpha2",        272, 2),
    ("logL M=32 std d=1056",     "results/sweep_v8_M32_loglinear",     1056, 2),
    ("logL M=32 chl. d=1056",    "results/sweep_v9_M32_challenging",   1056, 2),
    ("logL M=64 chl. d=4160",    "results/sweep_v9_M64_challenging",   4160, 2),
    ("NN M=16 d=6288",           "results/sweep_v8_M16_nn",            6288, 2),
]


def margin_for(sweep_dir: Path, N: int) -> Tuple[float, float, float]:
    """Returns (bai_mean, mlmc_mean, mlmc - bai). NaN if data missing."""
    if not sweep_dir.exists():
        return float("nan"), float("nan"), float("nan")
    runs = load_runs(sweep_dir)
    tab = plateau_table(runs)
    bai = tab.get(("bai_pg", N))
    mlmc = tab.get(("ganesh_mlmc", N))
    if bai is None or mlmc is None:
        return float("nan"), float("nan"), float("nan")
    return bai[0], mlmc[0], mlmc[0] - bai[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="figures/v9_summary/policy_effect_summary_v9.png")
    p.add_argument("--scatter_output", default="figures/v9_summary/margin_vs_d.png")
    args = p.parse_args()

    rows = []
    for label, dir_, d, N in SETTINGS:
        bai, mlmc, margin = margin_for(Path(dir_), N)
        rows.append((label, d, N, bai, mlmc, margin))
        print(f"{label:<28} d={d:<5} N={N:<3} bai={bai:>14.3f} mlmc={mlmc:>14.3f} margin={margin:>14.3f}")

    # ===== Bar chart =====
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    margins = np.array([r[5] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:green" if m > 0 else "tab:red" for m in margins]
    bars = ax.barh(range(len(rows)), margins, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("MLMC plateau − Bai-PG plateau  (positive = MLMC wins)")
    ax.set_title("MLMC-NPG win-margin across envs (positive = MLMC better)\n"
                 "x-axis: symlog; positive (green) = MLMC wins, red = MLMC loses")
    for i, (label, d, N, bai, mlmc, margin) in enumerate(rows):
        if not np.isnan(margin):
            ax.text(margin, i,
                    f"  N={N}, Δ={margin:+.2f}  (bai={bai:.1f}, mlmc={mlmc:.1f})",
                    fontsize=8, va="center")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")

    # ===== Margin-vs-d scatter =====
    fig, ax = plt.subplots(figsize=(9, 5))
    ds = [r[1] for r in rows if not np.isnan(r[5])]
    margs = [r[5] for r in rows if not np.isnan(r[5])]
    labs = [r[0] for r in rows if not np.isnan(r[5])]
    pos = np.array([m for m in margs if m > 0])
    pos_d = np.array([d for d, m in zip(ds, margs) if m > 0])
    neg = np.array([m for m in margs if m <= 0])
    neg_d = np.array([d for d, m in zip(ds, margs) if m <= 0])
    if len(pos) > 0:
        ax.scatter(pos_d, pos, c="tab:green", s=80, label="MLMC wins")
    if len(neg) > 0:
        ax.scatter(neg_d, np.abs(neg), c="tab:red", s=80, label="MLMC loses (|margin| plotted)")
    for d, m, lab in zip(ds, margs, labs):
        ax.annotate(lab, (d, abs(m) if m != 0 else 1), xytext=(5, 0),
                    textcoords="offset points", fontsize=7, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("d (number of policy parameters)")
    ax.set_ylabel("|MLMC plateau − Bai-PG plateau| (symlog)")
    ax.set_title("MLMC's win-margin vs policy-parameter count d\n"
                 "Each point: one (env, M, N) configuration")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out2 = Path(args.scatter_output)
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
