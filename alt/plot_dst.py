"""Plot Deep Sea Treasure 3-way sweep — paper B env 2.

Two-panel:
  (a) Convergence at focus_N (default 4): smoothed f vs iteration, mean+-std.
  (b) Plateau vs N: same axes as queueing N-sweep plots.

Also prints a per-algo plateau table (mean +- std + best-of-seeds).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SMOOTH_WINDOW = 20
COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG", "reinforce": r"REINFORCE ($\delta$=1)", "ganesh_mlmc": "MLMC-NPG"}
OPT_F = None  # filled in by main


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


def main(sweep_dir="results/sweep_dst",
         out_path="figures/dst/dst_3way.png",
         focus_N=4, opt_f=13.156):
    runs = load(sweep_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Convergence panel
    ax = axes[0]
    grouped = defaultdict(list)
    for r in runs:
        if r["config"]["N"] == focus_N:
            grouped[r["config"]["algorithm"]].append(r)
    for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
        rs = grouped.get(algo, [])
        if not rs: continue
        fs = [np.asarray(r["f"]) for r in rs]
        K = min(len(f) for f in fs)
        stack = np.stack([f[:K] for f in fs], axis=0)
        sm = trailing_mean(stack)
        m, s = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(K)
        ax.plot(x, m, color=COLOR[algo], label=LABEL[algo], lw=1.8)
        ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
    ax.axhline(opt_f, color="gray", linestyle="--", lw=1.0,
               label=f"f* (T4 deepest) = {opt_f:.2f}")
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$f_{\sqrt{\cdot}}(\hat J^{\pi_k})$")
    ax.set_title(f"Convergence at N={focus_N}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # Plateau-vs-N panel
    ax = axes[1]
    plats = defaultdict(list)
    for r in runs:
        c = r["config"]
        f = np.asarray(r["f"])
        last = f[-max(1, f.shape[0]//10):].mean()
        plats[(c["algorithm"], c["N"])].append(last)
    Ns = sorted({N for (_, N) in plats})
    for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
        xs, ms, ss = [], [], []
        for N in Ns:
            if (algo, N) in plats:
                vs = np.array(plats[(algo, N)])
                xs.append(N); ms.append(vs.mean()); ss.append(vs.std())
        if xs:
            ax.errorbar(xs, ms, yerr=ss, fmt="o-",
                        color=COLOR[algo], label=LABEL[algo],
                        lw=1.8, markersize=7, capsize=4)
    ax.axhline(opt_f, color="gray", linestyle="--", lw=1.0,
               label=f"f* deepest = {opt_f:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size $N$")
    ax.set_ylabel(r"Plateau $f_{\sqrt{\cdot}}$ (last 10% of K)")
    ax.set_title("Plateau vs. batch size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle(r"Deep Sea Treasure (Paper B env 2): $f(R, t) = \sqrt{R+\delta} + \sqrt{50-t+\delta}$, "
                 r"10x5 grid, tabular policy, $\gamma$=0.99, K=400, 5 seeds", fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # Print summary table
    print("\n== DST plateau summary (5 seeds) ==")
    print(f"{'N':>4} | "
          + " | ".join(f"{LABEL[a]:<24}" for a in ["bai_pg", "reinforce", "ganesh_mlmc"]))
    print("-" * 95)
    for N in Ns:
        row = f"{N:>4} | "
        for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
            if (algo, N) in plats:
                vs = np.array(plats[(algo, N)])
                marker = " *" if (max((np.mean(plats.get((a, N), [-1e30])) for a in ["bai_pg","reinforce","ganesh_mlmc"])) == vs.mean()) else "  "
                row += f"{vs.mean():>7.2f} +- {vs.std():>5.2f}{marker}    "
            else:
                row += f"{'—':>22}    "
        print(row)


if __name__ == "__main__":
    main()
