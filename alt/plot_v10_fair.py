"""Plot fair-budget comparisons: in each MLMC-dominant case, show
the original MLMC plateau alongside fair-budget Bai-PG and REINFORCE
in two modes (more K with same N, larger N with same K).

Reads:
  results/sweep_v9_<env>/                 — original MLMC reference data
  results/sweep_v10_fair_<env>/           — fair-budget Bai/REINFORCE
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


def plateau(f, frac=0.1):
    f = np.asarray(f)
    tail = max(1, int(len(f) * frac))
    return float(f[-tail:].mean())


def load_v10_fair(sweep_dir: Path):
    """Parse {algo}_{env}_N{N}_K{K}_{mode}_seed{seed}.json."""
    rows = []
    pat = re.compile(
        r"(?P<algo>bai_pg|reinforce)_"
        r"(?P<env>[a-z_]+?)_"
        r"N(?P<N>\d+)_K(?P<K>\d+)_"
        r"(?P<mode>moreK|largerN)_"
        r"seed(?P<seed>\d+)$")
    for fp in sorted(sweep_dir.glob("*.json")):
        m = pat.match(fp.stem)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        rows.append({
            "algo": m["algo"], "env": m["env"], "N": int(m["N"]),
            "K": int(m["K"]), "mode": m["mode"], "seed": int(m["seed"]),
            "f": d["f"], "trajectories": d["trajectories"],
            "plateau": plateau(d["f"]),
            "total_traj": d["trajectories"][-1],
        })
    return rows


def load_v9_ref(sweep_dir: Path, env_tag: str, target_Ns):
    rows = []
    for N in target_Ns:
        for fp in sorted(sweep_dir.glob(f"ganesh_mlmc_{env_tag}_N{N}_seed*.json")):
            with open(fp) as f:
                d = json.load(f)
            rows.append({
                "algo": "ganesh_mlmc", "env": env_tag, "N": N,
                "K": len(d["f"]), "mode": "ref",
                "seed": int(re.search(r"seed(\d+)", fp.stem).group(1)),
                "f": d["f"], "trajectories": d["trajectories"],
                "plateau": plateau(d["f"]),
                "total_traj": d["trajectories"][-1],
            })
    return rows


def aggregate_per_seed(rows, key):
    """Group rows by `key` tuple, return dict of (mean, std, n)."""
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r["plateau"])
    out = {}
    for k, vs in by.items():
        out[k] = (float(np.mean(vs)), float(np.std(vs)), len(vs))
    return out


def main_resource_or_dst(env_tag: str, label_envname: str, base_Ns: List[int]):
    fair_dir = Path(f"results/sweep_v10_fair_{env_tag}")
    ref_dir = Path(f"results/sweep_v9_{env_tag}")

    fair_rows = load_v10_fair(fair_dir)
    # Keep only matching env (filename tag may have "_" inside)
    fair_rows = [r for r in fair_rows if r["env"] == env_tag]

    ref_rows = load_v9_ref(ref_dir, env_tag, base_Ns)

    # MLMC reference per N
    print(f"\n=== {label_envname} ===")
    mlmc_per_N = aggregate_per_seed(ref_rows, lambda r: r["N"])
    for N in base_Ns:
        m = mlmc_per_N.get(N)
        if m:
            print(f"  MLMC N={N:<3}: plateau {m[0]:>10.3f} +- {m[1]:<8.3f}   (n={m[2]} seeds)")

    # Fair Bai/REINFORCE
    fair_per = aggregate_per_seed(
        fair_rows, lambda r: (r["algo"], r["N"], r["K"], r["mode"]))
    print(f"  Fair-budget Bai-PG / REINFORCE:")
    for (algo, N, K, mode), (mu, sigma, n) in sorted(fair_per.items()):
        print(f"    {algo:<10} N={N:<3} K={K:<5} {mode:<8}: plateau {mu:>10.3f} +- {sigma:<8.3f}   (n={n})")


def main_queue(label: str, mlmc_dir: str, fair_dir: str, M: int, target_N: int):
    """For queueing: read MLMC ref at N=target_N from mlmc_dir, and fair Bai/REINFORCE
    runs from fair_dir (which has run_sweep_v3 outputs)."""
    print(f"\n=== {label} ===")
    # MLMC reference
    pat = re.compile(rf"ganesh_mlmc_log_linear_M{M}_N{target_N}_alpha2\.0_seed(\d+)\.json")
    plats = []
    for fp in Path(mlmc_dir).glob("*.json"):
        m = pat.match(fp.name)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        plats.append(plateau(d["f"]))
    if plats:
        print(f"  MLMC N={target_N}: plateau {np.mean(plats):>14.2f} +- {np.std(plats):<10.2f}  (n={len(plats)})")

    # Fair Bai-PG / REINFORCE runs in fair_dir
    pat2 = re.compile(rf"(?P<algo>bai_pg|reinforce)_log_linear_M{M}_N(?P<N>\d+)_alpha2\.0_seed\d+\.json")
    by = defaultdict(list)
    by_K = {}
    for fp in Path(fair_dir).glob("*.json") if Path(fair_dir).exists() else []:
        m = pat2.match(fp.name)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        algo = m["algo"]
        N = int(m["N"])
        K = len(d["f"])
        by[(algo, N, K)].append(plateau(d["f"]))
        by_K[(algo, N, K)] = K
    print(f"  Fair-budget Bai-PG / REINFORCE:")
    for (algo, N, K), plats in sorted(by.items()):
        plats = np.asarray(plats)
        print(f"    {algo:<10} N={N:<3} K={K:<5}: plateau {plats.mean():>14.2f} +- {plats.std():<10.2f}  (n={len(plats)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["resource", "dst_large", "M32_std", "M32_chl", "M64_chl"])
    args = p.parse_args()

    if "resource" in args.cases:
        main_resource_or_dst("resource", "Resource Gathering (M=3)", [2, 16])
    if "dst_large" in args.cases:
        main_resource_or_dst("dst_large", "Larger DST 20x10 (d=804)", [4, 16])
    if "M32_std" in args.cases:
        main_queue("M=32 standard arrivals (d=1056)",
                    "results/sweep_v8_M32_loglinear",
                    "results/sweep_v10_fair_M32_std", 32, 2)
    if "M32_chl" in args.cases:
        main_queue("M=32 challenging arrivals (Sigma=0.8, d=1056)",
                    "results/sweep_v9_M32_challenging",
                    "results/sweep_v10_fair_M32_chl", 32, 2)
    if "M64_chl" in args.cases:
        main_queue("M=64 challenging arrivals (Sigma=0.8, d=4160)",
                    "results/sweep_v9_M64_challenging",
                    "results/sweep_v10_fair_M64_chl", 64, 2)


if __name__ == "__main__":
    main()
