"""Combined MLMC-showcase summary: 1x3 figure showing plateau-vs-N for
  (a) Deep Sea Treasure (paper B env 2, episodic, sqrt-fairness)
  (b) Purpose-built M=16 queueing (alpha=2, paper-style, lambda_min ~1/256)
  (c) Max-min fairness on M=8 queueing (Lipschitz, non-2nd-order-smooth)

Uses sign-normalized 'higher is better' axis on each panel for visual consistency.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plateau_stats(runs, sign=+1):
    out = defaultdict(list)
    for r in runs:
        c = r["config"]
        f = np.asarray(r["f"]) * sign
        last = f[-max(1, f.shape[0]//10):].mean()
        out[(c["algorithm"], c["N"])].append(last)
    return out


def panel(ax, sweep_dir, title, sign=+1, ylabel=None,
          ref_line=None, ref_label=None):
    runs = load(sweep_dir)
    plats = plateau_stats(runs, sign=sign)
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
    if ref_line is not None:
        ax.axhline(ref_line, color="gray", linestyle="--", lw=1.0,
                   label=ref_label or f"ref={ref_line:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size $N$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panel(
        axes[0],
        "results/sweep_dst",
        r"(a) Deep Sea Treasure — $\sqrt{R} + \sqrt{50-t}$, episodic",
        sign=+1,
        ylabel=r"Plateau $f_{\sqrt{\cdot}}$",
        ref_line=13.156,
        ref_label=r"$f^*$ deepest = 13.16",
    )

    panel(
        axes[1],
        "results/sweep_v7_M16_alpha2",
        r"(b) Purpose-built M=16 queueing, $\alpha$=2",
        sign=-1,
        ylabel=r"$-f_{\alpha=2}$ (lower better; flipped for vis)",
    )

    panel(
        axes[2],
        "results/sweep_v7_minfair_M8",
        r"(c) Max-min fairness M=8 — $\min_m J_m$",
        sign=+1,
        ylabel=r"Plateau $f_{\min} = H \cdot \min_m J_m$",
    )

    fig.suptitle("Three purpose-built MLMC-NPG showcase environments — plateau vs batch size",
                 fontsize=12)
    fig.tight_layout()
    out = Path("figures/v7_showcase/combined_3envs.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
