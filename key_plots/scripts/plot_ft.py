"""Regenerate the four ft_nn32_B16_NK40_a*_fair_budget plots.

Reads raw JSONs from ../data/fruit_tree/  (run from figures/key_plots/scripts/).
Writes PNGs and .tikz to ../  (i.e. figures/key_plots/).
"""
import json, re, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tikzplotlib

LABEL = {"ganesh_mlmc": "MLMC-NPG", "bai_pg": "Policy Gradient",
         "reinforce": "REINFORCE"}
COLOR = {"ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
         "reinforce": "tab:red"}
ORDER = list(LABEL.keys())

RUN_RE = re.compile(
    r"^(?P<algo>ganesh_mlmc|bai_pg|reinforce|scalarized_reinforce|linear_reinforce)_fruit_tree_"
    r"(?P<atag>a\d+)_d7_nn32_B16_N20_K(?P<K>\d+)_beta\S+_lr\S+_v15_ft_nn32_B16_NK40_"
    r"(?P<atag2>a\d+)_seed(?P<seed>\d+)\.json$"
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "fruit_tree"
OUT = HERE.parent

for atag, alpha in [("a010", "0.10"), ("a025", "0.25"), ("a050", "0.50"), ("a100", "1.00")]:
    runs = defaultdict(list)
    for fp in sorted(DATA.iterdir()):
        m = RUN_RE.match(fp.name)
        if not m or m.group("atag") != atag or m.group("algo") not in ORDER:
            continue
        d = json.load(open(fp))
        runs[m.group("algo")].append({
            "f": np.asarray(d["f"], float),
            "traj": np.asarray(d["trajectories"], int),
        })
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ORDER:
        rs = runs.get(algo, [])
        if not rs: continue
        t_lo = max(r["traj"][0] for r in rs); t_hi = min(r["traj"][-1] for r in rs)
        grid = np.linspace(t_lo, t_hi, 200)
        curves = np.stack([np.interp(grid, r["traj"], r["f"]) for r in rs])
        mean = curves.mean(axis=0); std = curves.std(axis=0, ddof=1)
        ax.plot(grid, mean, color=COLOR[algo], lw=1.7, label=f"{LABEL[algo]} (n={len(rs)})")
        ax.fill_between(grid, mean-std, mean+std, color=COLOR[algo], alpha=0.18)
    ax.set_xlabel("trajectories"); ax.set_ylabel("f(J)")
    ax.set_title(f"Fruit Tree (NN-32, B=16, N=20, K=40), alpha-fair alpha={alpha} -- 20 seeds, GPU", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / f"ft_nn32_B16_NK40_{atag}_fair_budget.png", dpi=150, bbox_inches="tight")
    tikzplotlib.save(str(OUT / f"ft_nn32_B16_NK40_{atag}_fair_budget.tikz"),
                     figure=fig, axis_height="6cm", axis_width="9cm")
    plt.close(fig)
    print(f"wrote {atag}")
