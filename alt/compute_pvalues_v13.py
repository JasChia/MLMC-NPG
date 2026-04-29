"""v13 Wilcoxon rank-sum p-values: MLMC-NPG vs each alternative.

Final-iteration f per seed, 2-sided exact MWU.

Sources:
  results/sweep_v13_dst_tab/                  6 DST variants, tabular
  results/sweep_v13_dst_nn32/                 6 DST variants, NN-32
  results/sweep_v13_fruit_tree_alphas/        5 alphas, tabular
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


REPO = Path(__file__).resolve().parents[1]


def load_finals_v13_dst(results_dir: Path, env: str, policy_tag: str) -> dict:
    """Returns {algo_label -> [final f over seeds]}."""
    out = defaultdict(list)
    for fp in sorted(results_dir.glob(f"*_{env}_{policy_tag}_*_v13_seed*.json")):
        if fp.name == "sweep_manifest.json":
            continue
        algo = fp.name.split("_" + env)[0]
        with open(fp) as f:
            d = json.load(f)
        out[algo].append(float(d["f"][-1]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def load_finals_v13_fruit_tree(results_dir: Path, alpha_tag: str) -> dict:
    """Returns {algo_label -> [final f over seeds]} for given alpha tag (e.g. '050')."""
    out = defaultdict(list)
    for fp in sorted(results_dir.glob(f"*_fruit_tree_a{alpha_tag}_tab_*_v13_seed*.json")):
        algo = fp.name.split("_fruit_tree")[0]
        with open(fp) as f:
            d = json.load(f)
        out[algo].append(float(d["f"][-1]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def ranksum_vs_mlmc(finals: dict, mlmc_key: str = "ganesh_mlmc") -> list:
    if mlmc_key not in finals:
        return []
    mlmc = finals[mlmc_key]
    rows = []
    for label, vals in finals.items():
        if label == mlmc_key:
            continue
        if len(vals) == 0 or len(mlmc) == 0:
            continue
        try:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided", method="exact")
        except ValueError:
            res = stats.mannwhitneyu(mlmc, vals, alternative="two-sided", method="asymptotic")
        rs = stats.ranksums(mlmc, vals, alternative="two-sided")
        rows.append({
            "alt": label,
            "n_M": len(mlmc), "n_X": len(vals),
            "M_mean": float(mlmc.mean()),
            "X_mean": float(vals.mean()),
            "M_sd": float(mlmc.std(ddof=1)) if len(mlmc) > 1 else 0.0,
            "X_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "diff": float(mlmc.mean() - vals.mean()),
            "p_mwu_exact": float(res.pvalue),
            "p_ranksum_z": float(rs.pvalue),
        })
    return rows


def print_rows(title: str, rows: list) -> None:
    print(f"\n## {title}")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'alt':<14} {'n_M':>3} {'n_X':>3} "
          f"{'M_mean':>9} {'X_mean':>9} {'M_sd':>7} {'X_sd':>7} "
          f"{'M-X':>8} {'p_MWU_exact':>12} {'p_rs_z':>9}")
    print("  " + "-" * 99)
    for r in rows:
        print(f"  {r['alt']:<14} {r['n_M']:>3d} {r['n_X']:>3d} "
              f"{r['M_mean']:>9.3f} {r['X_mean']:>9.3f} "
              f"{r['M_sd']:>7.3f} {r['X_sd']:>7.3f} "
              f"{r['diff']:>+8.3f} {r['p_mwu_exact']:>12.4f} "
              f"{r['p_ranksum_z']:>9.4f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dst_tab_dir", default="results/sweep_v13_dst_tab")
    p.add_argument("--dst_nn32_dir", default="results/sweep_v13_dst_nn32")
    p.add_argument("--fruit_tree_dir", default="results/sweep_v13_fruit_tree_alphas")
    p.add_argument("--label", default="v13",
                   help="label inserted in section headers, e.g. 'v13' or 'v13 (2K)'")
    args = p.parse_args()

    print("=" * 100)
    print(f"{args.label} rank-sum (MWU exact) MLMC-NPG vs alternatives")
    print("=" * 100)

    # --- DST tabular ---
    d_tab = Path(args.dst_tab_dir)
    if d_tab.exists():
        for env in ["dst_open", "dst_floor",
                    "dst_large_open", "dst_large_floor",
                    "dst_original_open", "dst_original_floor"]:
            finals = load_finals_v13_dst(d_tab, env, "tab")
            print_rows(f"DST {env}  (tabular, {args.label})", ranksum_vs_mlmc(finals))

    # --- DST NN-32 ---
    d_nn = Path(args.dst_nn32_dir)
    if d_nn.exists():
        for env in ["dst_open", "dst_floor",
                    "dst_large_open", "dst_large_floor",
                    "dst_original_open", "dst_original_floor"]:
            finals = load_finals_v13_dst(d_nn, env, "nn32")
            print_rows(f"DST {env}  (NN-32, {args.label})", ranksum_vs_mlmc(finals))

    # --- Fruit Tree alphas ---
    d_ft = Path(args.fruit_tree_dir)
    if d_ft.exists():
        for alpha_tag, alpha in [("000", 0.0), ("025", 0.25), ("050", 0.5),
                                  ("100", 1.0), ("200", 2.0)]:
            finals = load_finals_v13_fruit_tree(d_ft, alpha_tag)
            print_rows(f"Fruit Tree alpha={alpha}  (tabular, {args.label})",
                       ranksum_vs_mlmc(finals))


if __name__ == "__main__":
    main()
