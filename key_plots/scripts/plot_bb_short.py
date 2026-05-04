"""Regenerate bbs_T200_4algo_K40 plot.

Reads raw JSONs from ../data/bb_short/ (run from this script's directory).
Each file is a K=200 BB-short run; the curve is truncated to MLMC-NPG's K=40
cumulative-trajectory budget so all algorithms share the same horizontal axis.
Writes PNG and .tikz to ../.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

LABEL = {"ganesh_mlmc": "MLMC-NPG", "bai_pg": "Policy Gradient",
         "reinforce": "REINFORCE"}
COLOR = {"ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
         "reinforce": "tab:red"}
ORDER = list(LABEL.keys())

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "bb_short"
OUT = HERE.parent

def load(algo):
    runs = []
    for fp in sorted(DATA.iterdir()):
        if not fp.name.startswith(algo + "_"):
            continue
        d = json.load(open(fp))
        runs.append({"f": np.asarray(d["f"], float),
                     "traj": np.asarray(d["trajectories"], int)})
    return runs

runs = {a: load(a) for a in ORDER}
# Cap at MLMC iter 40 cumulative trajectories (averaged across seeds)
cap = float(np.mean([r["traj"][39] for r in runs["ganesh_mlmc"]]))

fig, ax = plt.subplots(figsize=(8, 5))
for algo in ORDER:
    rs = runs[algo]
    if not rs: continue
    tx, ty = [], []
    for r in rs:
        mask = r["traj"] <= cap
        tx.append(r["traj"][mask]); ty.append(r["f"][mask])
    t_lo = max(x[0] for x in tx if len(x) > 0)
    t_hi = min(x[-1] for x in tx if len(x) > 0)
    if t_lo >= t_hi: continue
    grid = np.linspace(t_lo, t_hi, 200)
    curves = np.stack([np.interp(grid, x, y) for x, y in zip(tx, ty)])
    mean = curves.mean(axis=0); std = curves.std(axis=0, ddof=1)
    ax.plot(grid, mean, color=COLOR[algo], lw=1.7, label=f"{LABEL[algo]} (n={len(rs)})")
    ax.fill_between(grid, mean-std, mean+std, color=COLOR[algo], alpha=0.18)

ax.set_xlabel("trajectories"); ax.set_ylabel("f(J)")
ax.set_title("BB-short T=200, K=40 (NN-32, B=8, N=20, gamma=0.9999), 20 seeds", fontsize=10)
ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "bbs_T200_4algo_K40.png", dpi=150, bbox_inches="tight")
tikzplotlib.save(str(OUT / "bbs_T200_4algo_K40.tikz"),
                 figure=fig, axis_height="6cm", axis_width="9cm")
print("wrote bbs_T200_4algo_K40")
