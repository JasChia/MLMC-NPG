"""Cross-case summary plot for v10 fair-budget experiments.
Bar chart: for each MLMC-dominant case, show:
  - headline margin (MLMC − Bai-PG at MLMC's small N)
  - fair-budget margin (MLMC − Bai-PG-largerN, matched per-iter trajectories)

Demonstrates how much of MLMC's "win" survives a fair sample budget.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plateau(f, frac=0.1):
    f = np.asarray(f)
    tail = max(1, int(len(f) * frac))
    return float(f[-tail:].mean())


def mean_plateau(dir_, pattern):
    plats = []
    for fp in Path(dir_).glob(pattern):
        with open(fp) as f:
            d = json.load(f)
        plats.append(plateau(d["f"]))
    return np.mean(plats) if plats else float("nan"), len(plats)


CASES = [
    # (label, d, target_N_mlmc, mlmc_dir, mlmc_pattern, headline_bai_dir, headline_bai_pattern, fair_bai_dir, fair_bai_pattern)
    ("Resource M=3\n(d=404, N=2)", 404, 2,
     "results/sweep_v9_resource",
     "ganesh_mlmc_resource_N2_seed*.json",
     "results/sweep_v9_resource",
     "bai_pg_resource_N2_seed*.json",
     "results/sweep_v10_fair_resource",
     "bai_pg_resource_N8_K300_largerN_seed*.json"),
    ("Resource M=3\n(d=404, N=16)", 404, 16,
     "results/sweep_v9_resource",
     "ganesh_mlmc_resource_N16_seed*.json",
     "results/sweep_v9_resource",
     "bai_pg_resource_N16_seed*.json",
     "results/sweep_v10_fair_resource",
     "bai_pg_resource_N44_K300_largerN_seed*.json"),
    ("Larger DST\n(d=804, N=4)", 804, 4,
     "results/sweep_v9_dst_large",
     "ganesh_mlmc_dst_large_N4_seed*.json",
     "results/sweep_v9_dst_large",
     "bai_pg_dst_large_N4_seed*.json",
     "results/sweep_v10_fair_dst_large",
     "bai_pg_dst_large_N13_K300_largerN_seed*.json"),
    ("Larger DST\n(d=804, N=16)", 804, 16,
     "results/sweep_v9_dst_large",
     "ganesh_mlmc_dst_large_N16_seed*.json",
     "results/sweep_v9_dst_large",
     "bai_pg_dst_large_N16_seed*.json",
     "results/sweep_v10_fair_dst_large",
     "bai_pg_dst_large_N44_K300_largerN_seed*.json"),
    ("M=32 std\n(d=1056, N=2)", 1056, 2,
     "results/sweep_v8_M32_loglinear",
     "ganesh_mlmc_log_linear_M32_N2_alpha2.0_seed*.json",
     "results/sweep_v8_M32_loglinear",
     "bai_pg_log_linear_M32_N2_alpha2.0_seed*.json",
     "results/sweep_v10_fair_M32_std_largerN",
     "bai_pg_log_linear_M32_N8_alpha2.0_seed*.json"),
    ("M=32 chl\n(d=1056, N=2)", 1056, 2,
     "results/sweep_v9_M32_challenging",
     "ganesh_mlmc_log_linear_M32_N2_alpha2.0_seed*.json",
     "results/sweep_v9_M32_challenging",
     "bai_pg_log_linear_M32_N2_alpha2.0_seed*.json",
     "results/sweep_v10_fair_M32_chl_largerN",
     "bai_pg_log_linear_M32_N8_alpha2.0_seed*.json"),
    ("M=64 chl\n(d=4160, N=2)", 4160, 2,
     "results/sweep_v9_M64_challenging",
     "ganesh_mlmc_log_linear_M64_N2_alpha2.0_seed*.json",
     "results/sweep_v9_M64_challenging",
     "bai_pg_log_linear_M64_N2_alpha2.0_seed*.json",
     "results/sweep_v10_fair_M64_chl_largerN",
     "bai_pg_log_linear_M64_N8_alpha2.0_seed*.json"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/v10_fair/v10_summary.png")
    args = p.parse_args()

    rows = []
    for label, d, N, m_dir, m_pat, hb_dir, hb_pat, fb_dir, fb_pat in CASES:
        mlmc_p, mlmc_n = mean_plateau(m_dir, m_pat)
        head_p, head_n = mean_plateau(hb_dir, hb_pat)
        fair_p, fair_n = mean_plateau(fb_dir, fb_pat)
        head_margin = mlmc_p - head_p if not (np.isnan(mlmc_p) or np.isnan(head_p)) else float("nan")
        fair_margin = mlmc_p - fair_p if not (np.isnan(mlmc_p) or np.isnan(fair_p)) else float("nan")
        rows.append({"label": label, "d": d, "N": N,
                     "mlmc": mlmc_p, "head_bai": head_p, "fair_bai": fair_p,
                     "head_margin": head_margin, "fair_margin": fair_margin,
                     "n_seeds": (mlmc_n, head_n, fair_n)})
        print(f"{label.replace(chr(10),' '):<30} mlmc={mlmc_p:>12.2f} head_bai={head_p:>14.2f} fair_bai={fair_p:>14.2f}  "
              f"head_margin={head_margin:>14.2f} fair_margin={fair_margin:>10.2f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(rows))
    width = 0.4
    head_margins = np.array([r["head_margin"] for r in rows])
    fair_margins = np.array([r["fair_margin"] for r in rows])
    ax.bar(xs - width/2, head_margins, width, color="tab:gray", alpha=0.7,
           label="Headline margin (MLMC vs Bai @ N_mlmc)")
    ax.bar(xs + width/2, fair_margins, width, color="tab:purple", alpha=0.7,
           label="Fair-budget margin (MLMC vs Bai-largerN, matched traj/iter)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("MLMC plateau − Bai-PG plateau  (positive = MLMC wins)")
    ax.set_title("Fair-budget reveals: many headline MLMC wins were sample-size artefacts\n"
                 "(positive = MLMC better; symlog y-axis)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    for i, r in enumerate(rows):
        if not np.isnan(r["head_margin"]):
            ax.annotate(f"{r['head_margin']:+.0f}" if abs(r['head_margin']) >= 1 else f"{r['head_margin']:+.2f}",
                        (i - width/2, r["head_margin"]), xytext=(0, 4),
                        textcoords="offset points", fontsize=7, ha="center")
        if not np.isnan(r["fair_margin"]):
            ax.annotate(f"{r['fair_margin']:+.0f}" if abs(r['fair_margin']) >= 1 else f"{r['fair_margin']:+.2f}",
                        (i + width/2, r["fair_margin"]), xytext=(0, 4),
                        textcoords="offset points", fontsize=7, ha="center")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
