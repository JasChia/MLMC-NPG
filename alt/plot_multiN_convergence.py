"""Multi-N convergence plotter: 1x3 panel showing convergence curves at
N=2, 4, 16 for a single env, all algorithms overlaid in each panel.

Use: python -m alt.plot_multiN_convergence <sweep_dir> <out_path> <env_label>
"""
import json
import sys
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


def plot_multiN(sweep_dir, out_path, title,
                Ns=(2, 4, 16), ylabel=r"$f$",
                ref_line=None, ref_label=None, sharey=True):
    runs = load(sweep_dir)
    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns) + 1, 4.7),
                             sharey=sharey)
    if len(Ns) == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        grouped = defaultdict(list)
        for r in runs:
            if r["config"]["N"] == N:
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
            ax.plot(x, m, color=COLOR[algo], label=LABEL[algo], lw=1.7)
            ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
        if ref_line is not None:
            ax.axhline(ref_line, color="gray", linestyle="--", lw=1.0,
                       label=ref_label)
        ax.set_xlabel("Outer iteration $k$")
        ax.set_title(f"N = {N}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    plot_multiN(
        "results/sweep_dst",
        "figures/dst/dst_3way_multiN.png",
        r"Deep Sea Treasure (Paper B env 2): convergence at N=2, 4, 16 — "
        r"$f(R, t) = \sqrt{R+\delta} + \sqrt{50-t+\delta}$",
        Ns=(2, 4, 16),
        ylabel=r"$f_{\sqrt{\cdot}}(\hat J^{\pi_k})$",
        ref_line=13.156, ref_label=r"$f^*$ deepest = 13.16",
        sharey=True,
    )
    plot_multiN(
        "results/sweep_v7_M16_alpha2",
        "figures/v7_showcase/m16_alpha2_multiN.png",
        r"M=16 queueing, $\alpha$=2: convergence at N=2, 4, 16",
        Ns=(2, 4, 16),
        ylabel=r"$f_{\alpha=2}(\hat J^{\pi_k})$",
        sharey=True,
    )
    plot_multiN(
        "results/sweep_v7_minfair_M8",
        "figures/v7_showcase/minfair_M8_multiN.png",
        r"M=8 max-min fairness ($f = H \cdot \min_m J_m$, "
        r"non-2nd-order-smooth): convergence at N=2, 4, 16",
        Ns=(2, 4, 16),
        ylabel=r"$f_{\min}(\hat J^{\pi_k})$",
        ref_line=5557.0, ref_label=r"arrival-limited optimum $\approx$ 5557",
        sharey=True,
    )
    plot_multiN(
        "results/sweep_v8_M32_loglinear",
        "figures/v8_largeM/m32_loglinear_multiN.png",
        r"M=32 queueing, $\alpha$=2 (log_linear policy, d=1056): convergence at N=2, 4, 16",
        Ns=(2, 4, 16),
        ylabel=r"$f_{\alpha=2}(\hat J^{\pi_k})$",
        sharey=True,
    )
    plot_multiN(
        "results/sweep_v8_M16_nn",
        "figures/v8_largeM/m16_nn_multiN.png",
        r"M=16 queueing, $\alpha$=2 (NN MLP 64x64, d=6288): convergence at N=2, 4 (K=100 due to NN cost)",
        Ns=(2, 4),
        ylabel=r"$f_{\alpha=2}(\hat J^{\pi_k})$",
        sharey=True,
    )


if __name__ == "__main__":
    main()
