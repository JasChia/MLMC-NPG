"""Dual-panel plot for large-M settings where Bai-PG diverges catastrophically:
  Top row: full y-range showing the divergence
  Bottom row: clipped y-range showing the stable methods' actual plateaus

Used for M=32 log_linear and M=16 NN where the bias-floor failure mode
(some queue J ~ 0 -> grad explodes) hits Bai-PG late in training.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SMOOTH_WINDOW = 15
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


def plot_dual(sweep_dir, out_path, title, Ns,
              clip_ylim=None, ylabel=r"$f$"):
    runs = load(sweep_dir)
    fig, axes = plt.subplots(2, len(Ns), figsize=(5 * len(Ns) + 1, 8.5),
                             sharex='col')
    if len(Ns) == 1:
        axes = axes[:, None]

    for col, N in enumerate(Ns):
        grouped = defaultdict(list)
        for r in runs:
            if r["config"]["N"] == N:
                grouped[r["config"]["algorithm"]].append(r)

        for row in (0, 1):
            ax = axes[row, col]
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

            if row == 0:
                ax.set_title(f"N = {N}  (full y-range)", fontsize=10)
            else:
                ax.set_title(f"N = {N}  (clipped y-range)", fontsize=10)
                if clip_ylim is not None:
                    ax.set_ylim(*clip_ylim)
            ax.set_xlabel("Outer iteration $k$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        axes[0, col].set_ylabel(ylabel) if col == 0 else None
        axes[1, col].set_ylabel(ylabel) if col == 0 else None

    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    plot_dual(
        "results/sweep_v8_M32_loglinear",
        "figures/v8_largeM/m32_loglinear_dual.png",
        r"M=32 queueing, $\alpha$=2 (log_linear, d=1056) — Bai-PG diverges late at small N; "
        r"MLMC-NPG stays stable",
        Ns=(2, 4, 16),
        clip_ylim=(-2500, -1500),
        ylabel=r"$f_{\alpha=2}(\hat J^{\pi_k})$",
    )
    plot_dual(
        "results/sweep_v8_M16_nn",
        "figures/v8_largeM/m16_nn_dual.png",
        r"M=16 queueing, $\alpha$=2 (NN MLP 64x64, d=6288) — Bai-PG diverges; "
        r"MLMC-NPG stays stable",
        Ns=(2, 4),
        clip_ylim=(-1200, -400),
        ylabel=r"$f_{\alpha=2}(\hat J^{\pi_k})$",
    )


if __name__ == "__main__":
    main()
