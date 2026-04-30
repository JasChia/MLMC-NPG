"""Plot MLMC margin vs alpha — find the curvature sweet spot for MLMC.

Combines alpha=1.0 from sweep_v6_logfair_Nsweep with alpha=1.5, 3.0 from
sweep_v6_alpha_curvature. All runs at M=8, N=4, K=300, log_linear, q_init=5,
N_inner=5, B_max=64 — apples-to-apples across alpha.

The reported quantity is the *normalized* MLMC margin over Bai-PG:

    margin(alpha) = (f_MLMC - f_Bai) / |grad scale at f|

since the absolute scale of f_alpha varies wildly with alpha.
For comparability across alphas, we instead report a *winning-fraction*
(across seeds) and the *seed-pair t-statistic*.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG", "reinforce": r"REINFORCE ($\delta$=1)", "ganesh_mlmc": "MLMC-NPG"}


def load_runs(dirs):
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                out.append(json.load(f))
    return out


def plateau_per_seed(runs, target_N=4, target_M=8):
    """Returns dict[(algo, alpha)] -> list of plateau-per-seed (sorted by seed)."""
    by_seed = defaultdict(dict)
    for r in runs:
        c = r["config"]
        if c.get("M") != target_M or c.get("N") != target_N:
            continue
        f = np.asarray(r["f"])
        last = f[-max(1, f.shape[0]//10):].mean()
        by_seed[(c["algorithm"], c["alpha"])][c["seed"]] = last
    out = {}
    for k, v in by_seed.items():
        out[k] = [v[s] for s in sorted(v.keys())]
    return out


def main(out_path="figures/v6_logfair/alpha_curvature_margin.png"):
    runs = load_runs([
        "results/sweep_v6_logfair_Nsweep",      # alpha=1.0
        "results/sweep_v6_alpha_curvature",     # alpha=1.5, 3.0
    ])
    plats = plateau_per_seed(runs, target_N=4, target_M=8)
    alphas = sorted({a for (_, a) in plats})
    algos = ["bai_pg", "reinforce", "ganesh_mlmc"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: per-seed plateaus, normalized so each alpha is centered at Bai mean.
    # That removes the alpha-dependent absolute scale of f.
    ax = axes[0]
    for algo in algos:
        xs, ys = [], []
        for alpha in alphas:
            if (algo, alpha) not in plats or ("bai_pg", alpha) not in plats:
                continue
            bai_mean = np.mean(plats[("bai_pg", alpha)])
            # margin per seed = (f_algo_seed - bai_mean) (sign convention: higher f better
            # for alpha=1 log; lower better for alpha>1 negative-power); normalize sign.
            sign = +1.0 if alpha == 1.0 else -1.0
            for v in plats[(algo, alpha)]:
                xs.append(alpha)
                ys.append(sign * (v - bai_mean))
        if xs:
            ax.scatter(xs, ys, color=COLOR[algo], label=LABEL[algo], alpha=0.7, s=60)
        # connect group means
        gm_xs, gm_ys = [], []
        for alpha in alphas:
            if (algo, alpha) in plats and ("bai_pg", alpha) in plats:
                bai_mean = np.mean(plats[("bai_pg", alpha)])
                sign = +1.0 if alpha == 1.0 else -1.0
                gm_xs.append(alpha)
                gm_ys.append(sign * (np.mean(plats[(algo, alpha)]) - bai_mean))
        if gm_xs:
            ax.plot(gm_xs, gm_ys, color=COLOR[algo], lw=1.4, alpha=0.6)
    ax.axhline(0, color="black", lw=0.7, linestyle="--",
               label="Bai-PG baseline")
    ax.set_xlabel(r"Scalarization $\alpha$  ($\alpha$=1: log; $\alpha$=2: paper A)")
    ax.set_ylabel(r"Margin over Bai-PG (sign-normalized; $>$0 is better)")
    ax.set_title("Per-seed margin vs scalarization")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="best")

    # Right panel: MLMC win-rate (fraction of seeds beating Bai-PG mean).
    ax = axes[1]
    for algo in ["reinforce", "ganesh_mlmc"]:
        xs, ws = [], []
        for alpha in alphas:
            if (algo, alpha) in plats and ("bai_pg", alpha) in plats:
                bai_mean = np.mean(plats[("bai_pg", alpha)])
                sign = +1.0 if alpha == 1.0 else -1.0
                seeds = plats[(algo, alpha)]
                wins = sum((sign * (v - bai_mean)) > 0 for v in seeds) / len(seeds)
                xs.append(alpha)
                ws.append(wins)
        if xs:
            ax.plot(xs, ws, "o-", color=COLOR[algo], label=LABEL[algo], lw=2,
                    markersize=10)
    ax.axhline(0.5, color="gray", lw=0.7, linestyle="--",
               label="Coin-flip vs Bai-PG")
    ax.set_xlabel(r"Scalarization $\alpha$")
    ax.set_ylabel("Fraction of seeds beating Bai-PG mean")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Win-rate against Bai-PG plateau (per seed)")
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle(r"MLMC's regime of dominance vs scalarization $\alpha$ "
                 r"(M=8, N=4, K=300, q$_0$=5, log_linear; 5 seeds at $\alpha$=1, 3 seeds at $\alpha$=1.5/3)",
                 fontsize=10.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # Print summary table
    print("\n== plateau means / per-seed values (M=8, N=4, K=300) ==")
    for alpha in alphas:
        print(f"\nalpha={alpha:.1f}")
        for algo in algos:
            if (algo, alpha) in plats:
                seeds = plats[(algo, alpha)]
                m = np.mean(seeds); s = np.std(seeds)
                seeds_str = " ".join(f"{v:.1f}" for v in seeds)
                print(f"  {algo:14s} mean={m:9.2f} +- {s:6.2f}  seeds=[{seeds_str}]")


if __name__ == "__main__":
    main()
