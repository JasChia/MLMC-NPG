"""Plot two purpose-built MLMC showcases:
  (a) M=16 queueing with alpha=2 — many objectives, small lambda, q0=5
  (b) M=8 queueing with max-min fairness (Lipschitz, non-2nd-order-smooth)

Each: 2-panel (convergence at focus_N + plateau-vs-N).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SMOOTH_WINDOW = 25
COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG", "reinforce": r"REINFORCE ($\delta$=1)", "ganesh_mlmc": "MLMC-NPG"}


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


def plot_one(sweep_dir, title, out_path, focus_N=4, ylabel=r"$f$", flip_sign=False):
    runs = load(sweep_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sign = -1 if flip_sign else +1

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
        stack = np.stack([f[:K] for f in fs], axis=0) * sign
        sm = trailing_mean(stack)
        m, s = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(K)
        ax.plot(x, m, color=COLOR[algo], label=LABEL[algo], lw=1.8)
        ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Convergence at N={focus_N}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes[1]
    plats = defaultdict(list)
    for r in runs:
        c = r["config"]
        f = np.asarray(r["f"])
        last = f[-max(1, f.shape[0]//10):].mean() * sign
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
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size $N$")
    ax.set_ylabel("Plateau " + ylabel)
    ax.set_title("Plateau vs. batch size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    print(f"\n== plateau summary: {title} ==")
    print(f"{'N':>4} | "
          + " | ".join(f"{LABEL[a]:<24}" for a in ["bai_pg", "reinforce", "ganesh_mlmc"]))
    print("-" * 95)
    for N in Ns:
        row = f"{N:>4} | "
        for algo in ["bai_pg", "reinforce", "ganesh_mlmc"]:
            if (algo, N) in plats:
                vs = np.array(plats[(algo, N)])
                row += f"{vs.mean():>8.2f} +- {vs.std():>5.2f}({len(vs)})    "
            else:
                row += f"{'—':>22}    "
        print(row)


if __name__ == "__main__":
    plot_one(
        "results/sweep_v7_M16_alpha2",
        r"M=16 queueing, $\alpha$=2, q$_0$=5: many objectives + sparse arrivals (showcase target)",
        "figures/v7_showcase/m16_alpha2.png",
        focus_N=4,
        ylabel=r"$f_{\alpha=2}$",
    )
    plot_one(
        "results/sweep_v7_minfair_M8",
        r"M=8 queueing, max-min fairness: $f(J) = H \cdot \min_m J_m$ (Lipschitz, non-2nd-order-smooth)",
        "figures/v7_showcase/minfair_M8.png",
        focus_N=4,
        ylabel=r"$f_{\min}$",
    )
