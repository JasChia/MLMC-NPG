"""Cross-experiment summary: MLMC's plateau-margin over Bai-PG, broken down by
(policy class, M) to expose the d-dependence of MLMC's advantage.

Loads from many sweep directories; each (policy, M) is a column with bars per N.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(d):
    out = []
    for p in sorted(Path(d).glob("*.json")):
        if p.name == "sweep_manifest.json":
            continue
        with open(p) as f:
            out.append(json.load(f))
    return out


def plateau_per_seed(runs, target_N, target_M=None, sign=+1):
    out = []
    for r in runs:
        c = r["config"]
        if c.get("N") != target_N:
            continue
        if target_M is not None and c.get("M") != target_M:
            continue
        f = np.asarray(r["f"]) * sign
        last = f[-max(1, f.shape[0]//10):].mean()
        out.append((c["algorithm"], last))
    return out


# Settings: (label, sweep_dir, M, sign — flip for negative-better f).
# All α=2 fairness; sign=+1 means "less negative is better"; we keep f as-is.
SETTINGS = [
    ("log_linear M=8 (d=72)",   "results/sweep_v5_loglinear_Nsweep", 8,  +1),
    ("tabular M=2 (d=242)",     "results/sweep_v5_tabular_Nsweep",   2,  +1),
    ("NN MLP M=8 (d=4482)",     "results/sweep_v5_nn_Nsweep",        8,  +1),
    ("log_linear M=16 (d=272)", "results/sweep_v7_M16_alpha2",       16, +1),
    ("log_linear M=32 (d=1056)","results/sweep_v8_M32_loglinear",    32, +1),
    ("NN MLP M=16 (d=6288)",    "results/sweep_v8_M16_nn",           16, +1),
]


def collect_margins():
    """For each setting, compute MLMC mean - Bai mean (per N)."""
    rows = []
    for label, sweep_dir, M, sign in SETTINGS:
        runs = load_runs(sweep_dir)
        if not runs:
            continue
        Ns = sorted({r["config"]["N"] for r in runs})
        for N in Ns:
            seed_data = plateau_per_seed(runs, target_N=N, target_M=M, sign=sign)
            algo_means = defaultdict(list)
            for algo, plat in seed_data:
                algo_means[algo].append(plat)
            if "bai_pg" not in algo_means or "ganesh_mlmc" not in algo_means:
                continue
            bai_mean = np.mean(algo_means["bai_pg"])
            mlmc_mean = np.mean(algo_means["ganesh_mlmc"])
            mlmc_std = np.std(algo_means["ganesh_mlmc"])
            bai_std = np.std(algo_means["bai_pg"])
            margin = mlmc_mean - bai_mean
            rows.append((label, N, margin, bai_std, mlmc_std,
                         len(algo_means["ganesh_mlmc"])))
    return rows


def main(out_path="figures/v8_largeM/policy_effect_summary.png"):
    rows = collect_margins()

    # Print table
    print(f"\n{'setting':<32} {'N':<4} {'MLMC-Bai margin':<16} {'(MLMC std)':<10} {'(Bai std)':<10} {'#seeds'}")
    print("-" * 95)
    for label, N, margin, bai_std, mlmc_std, nseeds in rows:
        print(f"{label:<32} {N:<4} {margin:>+14.3f}  {mlmc_std:<10.3f} {bai_std:<10.3f} {nseeds}")

    # Plot: grouped bars, one per (setting, N)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    settings_unique = []
    seen = set()
    for label, N, *_ in rows:
        if label not in seen:
            settings_unique.append(label); seen.add(label)
    n_settings = len(settings_unique)

    Ns_unique = sorted({N for _, N, *_ in rows})
    bar_width = 0.8 / len(Ns_unique)
    x = np.arange(n_settings)

    for i_n, N in enumerate(Ns_unique):
        ys = []
        es = []
        for label in settings_unique:
            match = [r for r in rows if r[0] == label and r[1] == N]
            if match:
                _, _, margin, _, mlmc_std, _ = match[0]
                ys.append(margin); es.append(mlmc_std)
            else:
                ys.append(np.nan); es.append(0)
        ax.bar(x + i_n * bar_width - 0.4 + bar_width/2, ys,
               width=bar_width, yerr=es, capsize=3,
               label=f"N={N}", edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_unique, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MLMC plateau − Bai-PG plateau (positive = MLMC wins)")
    ax.set_title("MLMC's win margin over Bai-PG vs (policy class, M) — bigger d => more MLMC advantage",
                 fontsize=11)
    ax.legend(title="Batch size", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("symlog", linthresh=1.0)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
