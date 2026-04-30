"""Generate convergence and bar-chart plots for v10 fair-budget experiments.

For each MLMC-dominant case, produce:
  - A convergence plot (cumulative trajectories on x-axis) overlaying:
      - MLMC-NPG reference (from v8/v9 sweeps)
      - Bai-PG fair-budget (moreK and largerN modes)
      - REINFORCE fair-budget
  - A bar chart of final plateaus (mean ± std across seeds)
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


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG", "reinforce": r"REINFORCE ($\delta$=1)",
         "ganesh_mlmc": "MLMC-NPG"}


def plateau(f, frac=0.1):
    f = np.asarray(f)
    tail = max(1, int(len(f) * frac))
    return float(f[-tail:].mean())


def load_v10_episodic(sweep_dir: Path, env_tag: str):
    """Load runs from results/sweep_v10_fair_<env>/ (episodic env naming)."""
    pat = re.compile(
        r"(?P<algo>bai_pg|reinforce)_"
        r"(?P<env>[a-z_]+?)_"
        r"N(?P<N>\d+)_K(?P<K>\d+)_"
        r"(?P<mode>moreK|largerN)_"
        r"seed(?P<seed>\d+)$")
    out = []
    for fp in sorted(sweep_dir.glob("*.json")):
        m = pat.match(fp.stem)
        if not m or m["env"] != env_tag:
            continue
        with open(fp) as f:
            d = json.load(f)
        out.append({
            "algo": m["algo"], "N": int(m["N"]), "K": int(m["K"]),
            "mode": m["mode"], "seed": int(m["seed"]),
            "f": d["f"], "trajectories": d["trajectories"],
        })
    return out


def load_v9_ref_episodic(sweep_dir: Path, env_tag: str, target_N: int):
    """Load MLMC reference from results/sweep_v9_<env>/ at the given N."""
    out = []
    for fp in sorted(sweep_dir.glob(f"ganesh_mlmc_{env_tag}_N{target_N}_seed*.json")):
        with open(fp) as f:
            d = json.load(f)
        out.append({
            "algo": "ganesh_mlmc", "N": target_N, "K": len(d["f"]),
            "mode": "ref",
            "seed": int(re.search(r"seed(\d+)", fp.stem).group(1)),
            "f": d["f"], "trajectories": d["trajectories"],
        })
    return out


def load_queueing(mlmc_dir: Path, fair_dirs: List[Path], M: int, target_N: int):
    """Returns (mlmc_runs, fair_runs)."""
    mlmc_runs = []
    pat_m = re.compile(rf"ganesh_mlmc_log_linear_M{M}_N{target_N}_alpha2\.0_seed(\d+)\.json")
    for fp in mlmc_dir.glob("*.json"):
        m = pat_m.match(fp.name)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        mlmc_runs.append({
            "algo": "ganesh_mlmc", "N": target_N, "K": len(d["f"]),
            "mode": "ref",
            "seed": int(m.group(1)),
            "f": d["f"], "trajectories": d["trajectories"],
        })

    fair_runs = []
    pat_f = re.compile(rf"(?P<algo>bai_pg|reinforce)_log_linear_M{M}_N(?P<N>\d+)_alpha2\.0_seed(?P<seed>\d+)\.json")
    for fd in fair_dirs:
        if not fd.exists():
            continue
        for fp in fd.glob("*.json"):
            m = pat_f.match(fp.name)
            if not m:
                continue
            with open(fp) as f:
                d = json.load(f)
            N = int(m["N"])
            K = len(d["f"])
            mode = "largerN" if N != target_N else "moreK"
            fair_runs.append({
                "algo": m["algo"], "N": N, "K": K, "mode": mode,
                "seed": int(m["seed"]),
                "f": d["f"], "trajectories": d["trajectories"],
            })
    return mlmc_runs, fair_runs


def aggregate_curves(runs):
    """Group by (algo, N, K, mode) and return mean/std per group."""
    by = defaultdict(list)
    for r in runs:
        by[(r["algo"], r["N"], r["K"], r["mode"])].append(r)
    out = {}
    for key, items in by.items():
        f_arrs = [np.asarray(r["f"]) for r in items]
        traj_arrs = [np.asarray(r["trajectories"]) for r in items]
        L = min(len(c) for c in f_arrs)
        f_stack = np.stack([c[:L] for c in f_arrs])
        traj_stack = np.stack([c[:L] for c in traj_arrs])
        out[key] = {
            "f_mean": f_stack.mean(axis=0),
            "f_std": f_stack.std(axis=0),
            "traj_mean": traj_stack.mean(axis=0),
            "n_seeds": len(items),
        }
    return out


def plot_case(ax_conv, ax_bar, title, mlmc_runs, fair_runs, target_N,
              mlmc_label="MLMC-NPG (ref)"):
    """Render convergence (left) + bar chart (right) for one case.
    Convergence: x=cumulative trajectories, y=f, mean±std bands.
    Bar: each algo+mode combo's plateau.
    """
    if not mlmc_runs:
        ax_conv.set_title(f"{title}\n(no MLMC reference data)")
        return

    # Aggregate curves
    mlmc_agg = aggregate_curves(mlmc_runs)
    fair_agg = aggregate_curves(fair_runs)

    # Convergence plot
    for key, agg in mlmc_agg.items():
        x = agg["traj_mean"]; y = agg["f_mean"]; ystd = agg["f_std"]
        ax_conv.plot(x, y, color=COLOR["ganesh_mlmc"], lw=2,
                     label=f"{mlmc_label} N={key[1]} (n={agg['n_seeds']})")
        ax_conv.fill_between(x, y - ystd, y + ystd,
                             color=COLOR["ganesh_mlmc"], alpha=0.2)

    # Order fair runs by (algo, mode)
    for key, agg in sorted(fair_agg.items()):
        algo, N, K, mode = key
        x = agg["traj_mean"]; y = agg["f_mean"]; ystd = agg["f_std"]
        c = COLOR[algo]
        ls = "--" if mode == "moreK" else ":"
        lw = 1.4
        lab = f"{LABEL[algo]} {mode} N={N} K={K} (n={agg['n_seeds']})"
        ax_conv.plot(x, y, color=c, ls=ls, lw=lw, label=lab)
        ax_conv.fill_between(x, y - ystd, y + ystd, color=c, alpha=0.1)

    ax_conv.set_xlabel("cumulative trajectories")
    ax_conv.set_ylabel("f(J)")
    ax_conv.set_title(title, fontsize=10)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(fontsize=7, loc="best")

    # Bar chart of final plateaus (last 10% mean)
    bars_data = []  # list of (label, mean, std, color)
    for key, agg in sorted(mlmc_agg.items()):
        L = len(agg["f_mean"])
        tail = max(1, L // 10)
        bars_data.append((f"MLMC N={key[1]}", agg["f_mean"][-tail:].mean(),
                          agg["f_std"][-tail:].mean(), COLOR["ganesh_mlmc"]))
    for key, agg in sorted(fair_agg.items()):
        algo, N, K, mode = key
        L = len(agg["f_mean"])
        tail = max(1, L // 10)
        lab = f"{LABEL[algo].split('(')[0].strip()}\n{mode} N={N}"
        bars_data.append((lab, agg["f_mean"][-tail:].mean(),
                          agg["f_std"][-tail:].mean(), COLOR[algo]))

    xs = np.arange(len(bars_data))
    means = np.array([b[1] for b in bars_data])
    stds = np.array([b[2] for b in bars_data])
    colors = [b[3] for b in bars_data]
    ax_bar.bar(xs, means, yerr=stds, color=colors, alpha=0.7, capsize=3)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([b[0] for b in bars_data], rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("plateau f(J)  (last 10% K mean)")
    ax_bar.grid(True, axis="y", alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="figures/v10_fair")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== Resource Gathering ======
    fair_resource = load_v10_episodic(Path("results/sweep_v10_fair_resource"), "resource")
    for target_N in [2, 16]:
        ref = load_v9_ref_episodic(Path("results/sweep_v9_resource"), "resource", target_N)
        # For fair runs, use ones whose trajectory budget matches the MLMC at this N
        # (filter by trajectory_total ~ MLMC's, OR by mode label since we know the matching pairs)
        # Simplest: pull all fair runs and let user see them. To avoid clutter, separate by base N.
        # For N=2 base: fair runs with N in {2, 8, 16} (since K=2 base has fair runs with K=1300/300/etc)
        # Actually distinguish by total trajectories: MLMC at N=2 ~ 5k, at N=16 ~ 25k. The fair runs
        # are designed to match one of those budgets each.
        if target_N == 2:
            # match Resource N=2 budget (~5k trajs): keep N in {2, 8} fair runs
            keep = [r for r in fair_resource if r["N"] in (2, 8)]
        else:
            # match Resource N=16 budget (~25k trajs): keep N in {16, 44, 86} fair runs
            keep = [r for r in fair_resource if r["N"] in (16, 44, 86)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [3, 2]})
        plot_case(axes[0], axes[1],
                  f"Resource Gathering (M=3, d=404) — fair-budget @ MLMC N={target_N} budget",
                  ref, keep, target_N)
        fig.tight_layout()
        out = out_dir / f"resource_fair_N{target_N}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")

    # ====== Larger DST ======
    fair_dst = load_v10_episodic(Path("results/sweep_v10_fair_dst_large"), "dst_large")
    for target_N in [4, 16]:
        ref = load_v9_ref_episodic(Path("results/sweep_v9_dst_large"), "dst_large", target_N)
        if target_N == 4:
            keep = [r for r in fair_dst if r["N"] in (4, 13, 26)]
        else:
            keep = [r for r in fair_dst if r["N"] in (16, 44, 86)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [3, 2]})
        plot_case(axes[0], axes[1],
                  f"Larger DST 20x10 (d=804) — fair-budget @ MLMC N={target_N} budget",
                  ref, keep, target_N)
        fig.tight_layout()
        out = out_dir / f"dst_large_fair_N{target_N}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")

    # ====== Queueing M=32 std / chl, M=64 chl ======
    queue_cases = [
        ("M=32 std (d=1056)",
         "results/sweep_v8_M32_loglinear",
         ["results/sweep_v10_fair_M32_std", "results/sweep_v10_fair_M32_std_largerN"],
         32, 2, "M32_std"),
        ("M=32 chl Σ=0.8 (d=1056)",
         "results/sweep_v9_M32_challenging",
         ["results/sweep_v10_fair_M32_chl", "results/sweep_v10_fair_M32_chl_largerN"],
         32, 2, "M32_chl"),
        ("M=64 chl Σ=0.8 (d=4160)",
         "results/sweep_v9_M64_challenging",
         ["results/sweep_v10_fair_M64_chl", "results/sweep_v10_fair_M64_chl_largerN"],
         64, 2, "M64_chl"),
    ]
    for title, mlmc_dir, fair_dirs, M, target_N, tag in queue_cases:
        mlmc_runs, fair_runs = load_queueing(
            Path(mlmc_dir), [Path(d) for d in fair_dirs], M, target_N)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [3, 2]})
        plot_case(axes[0], axes[1],
                  f"{title} — fair-budget @ MLMC N={target_N} budget",
                  mlmc_runs, fair_runs, target_N)
        # Use symlog on convergence y for queueing (Bai diverges to -10⁶)
        axes[0].set_yscale("symlog", linthresh=1000)
        axes[1].set_yscale("symlog", linthresh=1000)
        fig.tight_layout()
        out = out_dir / f"queue_{tag}_fair.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
