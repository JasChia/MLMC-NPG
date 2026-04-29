"""Less noisy Fruit Tree v11 plots using trailing running means over K.

For each run, f_k is replaced by the mean over the last `window` iterations
ending at k. Smoothed curves are then averaged across seeds.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {
    "bai_pg": "Bai-PG",
    "reinforce": r"REINFORCE ($\delta$=1)",
    "ganesh_mlmc": "MLMC-NPG",
}
POLICY_LABEL = {"tab": "tabular", "nn32": "NN-32", "nn128": "NN-128"}

RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc)_fruit_tree_"
    r"(?P<policy>tag_does_not_match|tab|nn32|nn128)_"
    r"N(?P<N>\d+)_K(?P<K>\d+)_(?P<mode>ref|moreK|largerN)_seed(?P<seed>\d+)$"
)


def running_mean(y: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    csum = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    for i in range(len(y)):
        start = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[start]) / (i + 1 - start)
    return out


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for sweep_dir in paths:
        if not sweep_dir.exists():
            continue
        for fp in sorted(sweep_dir.glob("*fruit_tree*.json")):
            if fp.name == "sweep_manifest.json":
                continue
            m = RUN_RE.match(fp.stem)
            if not m:
                continue
            with open(fp) as f:
                d = json.load(f)
            runs.append({
                "file": str(fp),
                "algo": m["algo"],
                "policy": m["policy"],
                "N": int(m["N"]),
                "K": int(m["K"]),
                "mode": m["mode"],
                "seed": int(m["seed"]),
                "f": np.asarray(d["f"], dtype=float),
                "trajectories": np.asarray(d["trajectories"], dtype=float),
            })
    return runs


def aggregate_smoothed(runs: list[dict], window: int) -> dict:
    by = defaultdict(list)
    for run in runs:
        by[(run["algo"], run["N"], run["K"], run["mode"])].append(run)

    out = {}
    for key, items in by.items():
        L = min(len(r["f"]) for r in items)
        f_stack = np.stack([running_mean(r["f"][:L], window) for r in items])
        x_stack = np.stack([r["trajectories"][:L] for r in items])
        out[key] = {
            "x_mean": x_stack.mean(axis=0),
            "f_mean": f_stack.mean(axis=0),
            "f_std": f_stack.std(axis=0),
            "n_seeds": len(items),
        }
    return out


def ordered_keys(agg: dict) -> list[tuple]:
    algo_rank = {"ganesh_mlmc": 0, "bai_pg": 1, "reinforce": 2}
    mode_rank = {"ref": 0, "moreK": 1, "largerN": 2}
    return sorted(agg.keys(), key=lambda k: (algo_rank[k[0]], mode_rank[k[3]], k[1], k[2]))


def plot_policy(runs: list[dict], policy: str, window: int, out: Path, x_axis: str) -> None:
    subset = [r for r in runs if r["policy"] == policy]
    if not subset:
        print(f"no runs for policy={policy}")
        return

    agg = aggregate_smoothed(subset, window)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for algo, N, K, mode in ordered_keys(agg):
        s = agg[(algo, N, K, mode)]
        color = COLOR[algo]
        ls = "-" if mode == "ref" else ("--" if mode == "moreK" else ":")
        label = f"{LABEL[algo]} {mode} N={N} K={K} (n={s['n_seeds']})"
        x = np.arange(len(s["f_mean"])) if x_axis == "iter" else s["x_mean"]
        ax.plot(x, s["f_mean"], color=color, ls=ls,
                lw=2.2 if algo == "ganesh_mlmc" else 1.6, label=label)
        ax.fill_between(
            x, s["f_mean"] - s["f_std"], s["f_mean"] + s["f_std"],
            color=color, alpha=0.12,
        )

    ax.set_title(f"Fruit Tree {POLICY_LABEL.get(policy, policy)}: trailing {window}-iteration mean")
    ax.set_xlabel("iteration K" if x_axis == "iter" else "cumulative trajectories")
    ax.set_ylabel("running mean f(J)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round_a", default="results/sweep_v11_round_a")
    p.add_argument("--round_b", default="results/sweep_v11_round_b")
    p.add_argument("--out_dir", default="figures/v11_fair_budget/running_mean")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--x_axis", choices=["iter", "trajectories"], default="trajectories")
    args = p.parse_args()

    runs = load_runs([Path(args.round_a), Path(args.round_b)])
    out_dir = Path(args.out_dir)
    for policy in ["tab", "nn32", "nn128"]:
        plot_policy(
            runs, policy, args.window,
            out_dir / f"fruit_tree_{policy}_running_mean_w{args.window}_{args.x_axis}.png",
            args.x_axis,
        )


if __name__ == "__main__":
    main()
