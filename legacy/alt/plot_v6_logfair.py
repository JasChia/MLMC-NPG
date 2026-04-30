"""Plot the alpha=1 (proportional fairness / log utility) showcase.

Mirrors plot_Nsweep.py but with axis labels for log fairness and an
"MLMC-NPG dominance regime" shaded band derived from the table.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 25

COLOR = {"bai_pg": "tab:blue",
         "reinforce": "tab:red",
         "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG",
         "reinforce": r"REINFORCE ($\delta$=1)",
         "ganesh_mlmc": "MLMC-NPG"}


def load(d):
    out = []
    for p in sorted(Path(d).glob("*.json")):
        if p.name == "sweep_manifest.json":
            continue
        with open(p) as f:
            out.append(json.load(f))
    return out


def trailing_mean(stack, w=SMOOTH_WINDOW):
    K = stack.shape[1]; w = min(w, K)
    cs = np.cumsum(stack, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    lo = np.maximum(0, np.arange(K) - w + 1); hi = np.arange(K) + 1
    counts = (hi - lo).astype(np.float64)
    return (cs[:, hi] - cs[:, lo]) / counts[None, :]


def plateau_stats(runs):
    out = defaultdict(list)
    for r in runs:
        c = r["config"]
        f = np.asarray(r["f"])
        last = f[-max(1, f.shape[0]//10):].mean()
        out[(c["algorithm"], c["N"])].append(last)
    summary = {}
    for key, vals in out.items():
        vals = np.array(vals)
        summary[key] = (vals.mean(), vals.std(), len(vals))
    return summary


def plot_logfair(sweep_dir, out_path, focus_N=4):
    runs = load(sweep_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    # Left: convergence curves at focus_N (log-utility f goes UP → max).
    focus = [r for r in runs if r["config"]["N"] == focus_N]
    grouped = defaultdict(list)
    for r in focus:
        grouped[r["config"]["algorithm"]].append(r)
    for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
        rs = grouped.get(algo, [])
        if not rs:
            continue
        fs = [np.asarray(r["f"]) for r in rs]
        K = min(len(f) for f in fs)
        stack = np.stack([f[:K] for f in fs], axis=0)
        sm = trailing_mean(stack)
        m, s = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(K)
        ax1.plot(x, m, color=COLOR[algo], label=LABEL[algo], lw=1.8)
        ax1.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
    ax1.set_xlabel("Outer iteration $k$")
    ax1.set_ylabel(r"$f_{\alpha=1}(\hat J^{\pi_k}) = H \sum_m \log(\hat J_m + \delta)$")
    ax1.set_title(f"Convergence at N={focus_N}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", frameon=False)

    # Right: plateau vs N.
    plats = plateau_stats(runs)
    Ns_all = sorted({N for (_, N) in plats})
    # detect MLMC-winning N values for shading
    winning_Ns = []
    for N in Ns_all:
        scores = []
        for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
            if (algo, N) in plats:
                scores.append((algo, plats[(algo, N)][0]))
        if scores and max(scores, key=lambda x: x[1])[0] == "ganesh_mlmc":
            winning_Ns.append(N)

    for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
        xs, ms, ss = [], [], []
        for N in Ns_all:
            if (algo, N) in plats:
                m, s, _ = plats[(algo, N)]
                xs.append(N); ms.append(m); ss.append(s)
        if xs:
            ax2.errorbar(xs, ms, yerr=ss, fmt="o-",
                         color=COLOR[algo], label=LABEL[algo],
                         lw=1.8, markersize=7, capsize=4)
    if winning_Ns:
        for wN in winning_Ns:
            ax2.axvspan(wN / np.sqrt(2), wN * np.sqrt(2),
                        color="tab:purple", alpha=0.10)
        # one legend entry for the band
        ax2.axvspan(winning_Ns[0] / np.sqrt(2), winning_Ns[0] / np.sqrt(2),
                    color="tab:purple", alpha=0.10,
                    label="MLMC-NPG wins (observed)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch size $N$")
    ax2.set_ylabel(r"Plateau $f_{\alpha=1}$ (last 10% of $K$)")
    ax2.set_title("Plateau vs. batch size")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=False)

    fig.suptitle(r"log-utility (proportional fairness, $\alpha$=1) "
                 r"on M=8 queueing, q$_0$=5, paper env 3 scalarization",
                 fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # Print summary table
    print("\n== alpha=1 plateau summary (M=8 log_linear, q0=5) ==")
    print(f"{'N':>4} | "
          + " | ".join(f"{LABEL[a]:<22}" for a in ["bai_pg", "reinforce", "ganesh_mlmc"]))
    print("-" * 90)
    for N in Ns_all:
        row = f"{N:>4} | "
        for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
            if (algo, N) in plats:
                m, s, n = plats[(algo, N)]
                marker = " *" if (max((plats.get((a, N), (-1e30,))[0] for a in ["bai_pg","reinforce","ganesh_mlmc"])) == m) else "  "
                row += f"{m:>9.2f} +- {s:>5.2f}{marker} "
            else:
                row += f"{'—':>20}  "
        print(row)


if __name__ == "__main__":
    import sys
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_v6_logfair_Nsweep"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "figures/v6_logfair/loglinear_M8_alpha1.png"
    plot_logfair(sweep_dir, out_path)
