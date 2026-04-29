"""Plot MLMC-NPG vs Bai-PG plateau as B varies, at the winning Fruit Tree NN-32 a=0 config."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIGS = [
    # (tag, B, n_seeds_label)
    ("ft_nn32_B4_NK20_a000", 4),
    ("ft_nn32_B8_NK20_a000", 8),
    ("ft_nn32_B16_NK20_a000", 16),
]
ALGOS = ["ganesh_mlmc", "bai_pg", "reinforce", "scalarized_reinforce"]
LABEL = {
    "ganesh_mlmc": "MLMC-NPG", "bai_pg": "Bai-PG",
    "reinforce": "REINFORCE (paper A)",
    "scalarized_reinforce": "REINFORCE (scalarized)",
}
COLOR = {
    "ganesh_mlmc": "tab:purple", "bai_pg": "tab:blue",
    "reinforce": "tab:red", "scalarized_reinforce": "tab:orange",
}


def load_finals(results_dir: Path, tag: str) -> dict:
    """Returns {algo -> array of final f}."""
    out = {a: [] for a in ALGOS}
    for fp in sorted(results_dir.glob(f"*_v14_{tag}_seed*.json")):
        algo = next(a for a in ALGOS if fp.name.startswith(a + "_"))
        with open(fp) as f:
            d = json.load(f)
        out[algo].append(float(d["f"][-1]))
    return {a: np.asarray(v, dtype=np.float64) for a, v in out.items()}


def main() -> None:
    results_dir = Path("results/sweep_v14")
    means = {a: [] for a in ALGOS}
    sds = {a: [] for a in ALGOS}
    Bs = []
    for tag, B in CONFIGS:
        finals = load_finals(results_dir, tag)
        Bs.append(B)
        for a in ALGOS:
            v = finals[a]
            means[a].append(float(v.mean()) if len(v) else np.nan)
            sds[a].append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ALGOS:
        m = np.array(means[a]); s = np.array(sds[a])
        ax.errorbar(Bs, m, yerr=s, color=COLOR[a], marker="o",
                    capsize=4, linewidth=1.6, markersize=7,
                    label=LABEL[a])
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("inner batch B (= MLMC B_max)")
    ax.set_ylabel("plateau f(J)  (mean ± sd, 7 seeds)")
    ax.set_title("Fruit Tree NN-32, alpha=0 (linear): plateau vs B  (N=K=20, fair budget)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = Path("figures/v14/B_scaling_a000.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
