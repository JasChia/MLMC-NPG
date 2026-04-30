"""Summary visualizations of the v5 multi-N, multi-policy findings.

Two figures:
  1. plateau_vs_N_3policies.png — 3-panel side-by-side showing plateau
     vs N for log-linear / tabular / NN, with the MLMC-winning regime shaded.
  2. nn_tricks_helps_hurts.png — grouped bar chart showing that
     lr=0.001 + grad_clip=10 rescues Bai-PG/REINFORCE but breaks MLMC.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load(path):
    out = []
    for p in sorted(Path(path).glob("*.json")):
        if p.name == "sweep_manifest.json":
            continue
        with open(p) as f:
            out.append(json.load(f))
    return out


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


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {"bai_pg": "Bai-PG", "reinforce": r"REINFORCE $\delta$=1", "ganesh_mlmc": "MLMC-NPG"}


# ============================================================
# Figure 1: plateau vs N for all 3 policy classes side by side
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

panels = [
    ("log_linear (M=8, d=72)", "results/sweep_v5_loglinear_Nsweep",
     axes[0]),
    ("tabular (M=2, d=242)", "results/sweep_v5_tabular_Nsweep",
     axes[1]),
    ("NN MLP 64x64 (M=8, d=4482)", "results/sweep_v5_nn_Nsweep",
     axes[2]),
]

for title, path, ax in panels:
    runs = load(path)
    plats = plateau_stats(runs)
    algos = ["bai_pg", "reinforce", "ganesh_mlmc"]
    Ns_all = sorted({N for (_, N) in plats})
    for algo in algos:
        xs, means, stds = [], [], []
        for N in Ns_all:
            if (algo, N) in plats:
                m, s, _ = plats[(algo, N)]
                # Cap crazy outliers (NN REINFORCE at N=4 exploded)
                if abs(m) > 1e4:
                    continue
                xs.append(N); means.append(m); stds.append(s)
        if xs:
            xs = np.array(xs); means = np.array(means); stds = np.array(stds)
            ax.errorbar(xs, means, yerr=stds, fmt="o-",
                        color=COLOR[algo], label=LABEL[algo],
                        lw=1.8, markersize=7, capsize=4)
    # Shade the "MLMC wins" region, identified from plateau_vs_N for each policy
    # These shading thresholds are manual observations:
    if "log_linear" in title:
        ax.axvspan(1.5, 6, color="tab:purple", alpha=0.08,
                   label="MLMC-NPG wins (observed)")
    elif "tabular" in title:
        # MLMC wins at N=2, 16, 256 on tabular (not N=4)
        ax.axvspan(1.5, 3, color="tab:purple", alpha=0.08,
                   label="MLMC-NPG wins (observed)")
        ax.axvspan(10, 400, color="tab:purple", alpha=0.08)
    else:  # NN
        ax.axvspan(1.5, 20, color="tab:purple", alpha=0.08,
                   label="MLMC-NPG wins (observed)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size $N$")
    ax.set_ylabel(r"Plateau $f_{\alpha=2}$" if ax is axes[0] else "")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

fig.suptitle("MLMC-NPG's regime of dominance varies with policy class "
             r"(q$_0$=5, M varies, $\alpha$=2, K=500)",
             fontsize=12)
fig.tight_layout()
out = Path("figures/v5_Nsweep/plateau_vs_N_3policies.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved {out}")


# ============================================================
# Figure 2: NN tricks helps/hurts bar chart (N=4 only)
# ============================================================

runs_nn = load("results/sweep_v5_nn_Nsweep")
runs_nn_tricks = load("results/sweep_v5_nn_tricks")

# Filter to N=4
get = lambda runs, algo: [r for r in runs
                          if r["config"]["algorithm"] == algo
                          and r["config"]["N"] == 4]

def plateau(rs):
    if not rs:
        return None, None
    vals = [np.asarray(r["f"])[-max(1, len(r["f"])//10):].mean() for r in rs]
    return np.mean(vals), np.std(vals)

fig, ax = plt.subplots(figsize=(9, 5))
algos = ["bai_pg", "reinforce", "ganesh_mlmc"]
x = np.arange(len(algos))
width = 0.35

means_plain = []; stds_plain = []
means_tricks = []; stds_tricks = []
for algo in algos:
    m1, s1 = plateau(get(runs_nn, algo))
    m2, s2 = plateau(get(runs_nn_tricks, algo))
    # Cap obvious outliers for visualization
    m1_show = max(m1, -300) if m1 is not None else np.nan
    s1_show = min(s1, 100) if s1 is not None else 0
    means_plain.append(m1_show); stds_plain.append(s1_show)
    means_tricks.append(m2); stds_tricks.append(s2)

bars1 = ax.bar(x - width/2, means_plain, width, yerr=stds_plain,
               label="Plain NN (lr=0.005, no clip)", color="tab:cyan",
               edgecolor="black", capsize=5)
bars2 = ax.bar(x + width/2, means_tricks, width, yerr=stds_tricks,
               label="NN + tricks (lr=0.001, grad_clip=10)",
               color="tab:orange", edgecolor="black", capsize=5)

# Annotate REINFORCE plain with "catastrophic -10^6"
ax.annotate("catastrophic\n(−10⁶, clipped for vis)",
            xy=(1 - width/2, -300), xytext=(0.6, -250),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

# Annotate MLMC tricks with "breaks it"
ax.annotate(r"tricks break MLMC",
            xy=(2 + width/2, means_tricks[2]),
            xytext=(1.9, -220),
            fontsize=9, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))

ax.set_xticks(x)
ax.set_xticklabels([LABEL[a] for a in algos], fontsize=10)
ax.set_ylabel(r"Plateau $f_{\alpha=2}$ at N=4 (M=8, NN policy)")
ax.set_title("NN-stabilization tricks: help PG/REINFORCE, hurt MLMC-NPG")
ax.legend(loc="lower right", frameon=False)
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylim(-310, -110)

fig.tight_layout()
out = Path("figures/v5_Nsweep/nn_tricks_helps_hurts.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved {out}")
