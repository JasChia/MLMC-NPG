"""Regenerate ft_alpha_sweep_3algo.tikz as a 2x2 groupplot.

Reads reinforce/ganesh_mlmc/bai_pg JSONs from ../data/fruit_tree/.
Writes ft_alpha_sweep_3algo.tikz (and a companion PNG) to ../.
Run from key_plots/scripts/.
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tikzplotlib

LABEL = {"ganesh_mlmc": "MLMC-NPG", "bai_pg": "Policy Gradient",
         "reinforce": "REINFORCE"}
COLOR = {"ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
         "reinforce": "tab:red"}
ORDER = list(LABEL.keys())

ALPHAS = [("a010", "0.10"), ("a025", "0.25"), ("a050", "0.50"), ("a100", "1.00")]
PANEL  = ["(a)", "(b)", "(c)", "(d)"]

RUN_RE = re.compile(
    r"^(?P<algo>ganesh_mlmc|bai_pg|reinforce|scalarized_reinforce|linear_reinforce)"
    r"_fruit_tree_(?P<atag>a\d+)_d7_nn32_B16_N20_K(?P<K>\d+)_beta\S+_lr\S+"
    r"_v15_ft_nn32_B16_NK40_(?P<atag2>a\d+)_seed(?P<seed>\d+)\.json$"
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "fruit_tree"
OUT  = HERE.parent

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for idx, ((atag, alpha), panel) in enumerate(zip(ALPHAS, PANEL)):
    runs = defaultdict(list)
    for fp in sorted(DATA.iterdir()):
        m = RUN_RE.match(fp.name)
        if not m or m.group("atag") != atag or m.group("algo") not in ORDER:
            continue
        d = json.load(open(fp))
        runs[m.group("algo")].append({
            "f":    np.asarray(d["f"], float),
            "traj": np.asarray(d["trajectories"], int),
        })

    ax = axes_flat[idx]
    for algo in ORDER:
        rs = runs.get(algo, [])
        if not rs:
            continue
        t_lo = max(r["traj"][0]  for r in rs)
        t_hi = min(r["traj"][-1] for r in rs)
        grid   = np.linspace(t_lo, t_hi, 200)
        curves = np.stack([np.interp(grid, r["traj"], r["f"]) for r in rs])
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0, ddof=1)
        ax.plot(grid, mean, color=COLOR[algo], lw=1.7,
                label=f"{LABEL[algo]} (n={len(rs)})")
        ax.fill_between(grid, mean - std, mean + std,
                        color=COLOR[algo], alpha=0.18)

    ax.set_title(rf"\textbf{{{panel}}} $\alpha = {alpha}$")
    ax.set_xlabel("Sampled Trajectories")
    if idx % 2 == 0:
        ax.set_ylabel(r"Fruit Tree: $f(\mathbf{J}_\pi)$")
    ax.tick_params(axis="both", direction="out")
    if idx == 0:
        ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "ft_alpha_sweep_3algo.png", dpi=150, bbox_inches="tight")

tikzplotlib.save(
    str(OUT / "ft_alpha_sweep_3algo.tikz"),
    figure=fig,
    axis_height="6cm",
    axis_width="7cm",
    extra_axis_parameters=[
        "xtick={0,4000,8000,12000}",
        "scaled x ticks=false",
    ],
)
plt.close(fig)
print("wrote ft_alpha_sweep_3algo.tikz")
