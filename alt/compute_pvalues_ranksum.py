"""Wilcoxon rank-sum (Mann-Whitney U) p-values: MLMC vs each other method.

Final-iteration scalarized objective `f` per seed is the test statistic.
Two-sided test, with effect-direction noted (MLMC mean - other mean).

Envs:
  - 10x5 DST: dst_open, dst_floor (results/sweep_v12_dst_open_variants_..._combined)
  - Large DST: dst_large_open, dst_large_floor (results/sweep_v12_dst_traj_..._combined)
  - Fruit Tree: fruit_tree_tab (round_a), fruit_tree_nn32 (round_b)
    (only 3 seeds available for these — flagged in output)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


REPO = Path(__file__).resolve().parents[1]


METHODS = [
    # (label, algo, N, K, mode)
    ("MLMC ref",         "ganesh_mlmc", 4,  300,  "ref"),
    ("Bai moreK",        "bai_pg",      4,  1000, "moreK"),
    ("Bai largerN",      "bai_pg",      13, 300,  "largerN"),
    ("REINFORCE moreK",  "reinforce",   4,  2000, "moreK"),
    ("REINFORCE largerN","reinforce",   26, 300,  "largerN"),
]


def load_finals(results_dir: Path, env: str, policy_tag: str, seeds):
    """Return dict[label] -> array of final f values, one per seed."""
    out = {}
    for label, algo, N, K, mode in METHODS:
        vals = []
        for s in seeds:
            f = results_dir / f"{algo}_{env}_{policy_tag}_N{N}_K{K}_{mode}_seed{s}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                h = json.load(fh)
            vals.append(float(h["f"][-1]))
        out[label] = np.asarray(vals, dtype=np.float64)
    return out


def ranksum_vs_mlmc(finals: dict):
    mlmc = finals["MLMC ref"]
    rows = []
    for label, vals in finals.items():
        if label == "MLMC ref":
            continue
        if len(vals) == 0 or len(mlmc) == 0:
            rows.append((label, len(mlmc), len(vals), np.nan, np.nan,
                         float(mlmc.mean()) if len(mlmc) else np.nan,
                         float(vals.mean()) if len(vals) else np.nan,
                         np.nan, np.nan, np.nan))
            continue
        # mannwhitneyu == Wilcoxon rank-sum; use exact for small n, two-sided.
        try:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided",
                                      method="exact")
        except ValueError:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided",
                                      method="asymptotic")
        # also report scipy.stats.ranksums for the asymptotic-z variant
        rs = stats.ranksums(mlmc, vals, alternative="two-sided")
        rows.append((label, len(mlmc), len(vals),
                     float(mlmc.mean()), float(vals.mean()),
                     float(mlmc.std(ddof=1)) if len(mlmc) > 1 else 0.0,
                     float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                     float(mlmc.mean() - vals.mean()),
                     float(res.pvalue), float(rs.pvalue)))
    return rows


def print_table(title: str, rows):
    print(f"\n## {title}")
    print(f"{'method':<20} {'n_M':>4} {'n_X':>4} "
          f"{'MLMC mean':>10} {'other mean':>11} "
          f"{'MLMC sd':>9} {'other sd':>9} "
          f"{'M-X':>9} {'p MWU exact':>12} {'p ranksum z':>12}")
    print("-" * 117)
    for r in rows:
        (label, nM, nX, mM, mX, sM, sX, d, p_mwu, p_rs) = r
        if np.isnan(p_mwu):
            print(f"{label:<20} {nM:>4d} {nX:>4d} "
                  f"{mM:>10.3f} {mX:>11.3f} "
                  f"{sM:>9.3f} {sX:>9.3f} "
                  f"{d:>9.3f} {'n/a':>12} {'n/a':>12}")
        else:
            print(f"{label:<20} {nM:>4d} {nX:>4d} "
                  f"{mM:>10.3f} {mX:>11.3f} "
                  f"{sM:>9.3f} {sX:>9.3f} "
                  f"{d:>+9.3f} {p_mwu:>12.4f} {p_rs:>12.4f}")


def main():
    seeds10 = list(range(3, 13))
    seeds3 = [0, 1, 2]

    # --- 10x5 DST open / floor (v12, 10 seeds 3-12) ---
    d_small = REPO / "results" / "sweep_v12_dst_open_variants_gamma09999_seed3_12_reinforce_fixed_combined"
    print_table("10x5 DST open  (v12, gamma=0.9999, seeds 3-12, tabular)",
                ranksum_vs_mlmc(load_finals(d_small, "dst_open", "tab", seeds10)))
    print_table("10x5 DST floor (v12, gamma=0.9999, seeds 3-12, tabular)",
                ranksum_vs_mlmc(load_finals(d_small, "dst_floor", "tab", seeds10)))

    # --- Large DST open / floor (v12, 10 seeds 3-12) ---
    d_large = REPO / "results" / "sweep_v12_dst_traj_gamma09999_seed3_12_reinforce_fixed_combined"
    print_table("Large DST open  (v12, gamma=0.9999, seeds 3-12, tabular)",
                ranksum_vs_mlmc(load_finals(d_large, "dst_large_open", "tab", seeds10)))
    print_table("Large DST floor (v12, gamma=0.9999, seeds 3-12, tabular)",
                ranksum_vs_mlmc(load_finals(d_large, "dst_large_floor", "tab", seeds10)))

    # --- Fruit Tree tabular (v11 round_a, only 3 seeds) ---
    d_ft_tab = REPO / "results" / "sweep_v11_round_a"
    print_table("Fruit Tree tabular (v11 round_a, seeds 0-2)  ** only 3 seeds available **",
                ranksum_vs_mlmc(load_finals(d_ft_tab, "fruit_tree", "tab", seeds3)))

    # --- Fruit Tree NN-32 (v11 round_b, only 3 seeds) ---
    d_ft_nn = REPO / "results" / "sweep_v11_round_b"
    print_table("Fruit Tree NN-32 (v11 round_b, seeds 0-2)  ** only 3 seeds available **",
                ranksum_vs_mlmc(load_finals(d_ft_nn, "fruit_tree", "nn32", seeds3)))


if __name__ == "__main__":
    main()
