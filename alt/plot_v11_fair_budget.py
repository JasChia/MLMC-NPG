"""Plots and summary tables for v11 fair-budget experiments.

Input filename schema:
  {algo}_{env}_{policy_tag}_N{N}_K{K}_{mode}_seed{seed}.json

Outputs:
  - per-env convergence + plateau panels for Round A
  - per-policy convergence + plateau panels for Round B
  - cross-env margin summary
  - Fruit Tree d-scale summary
  - CSV tables used by REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


COLOR = {"bai_pg": "tab:blue", "reinforce": "tab:red", "ganesh_mlmc": "tab:purple"}
LABEL = {
    "bai_pg": "Bai-PG",
    "reinforce": r"REINFORCE ($\delta$=1)",
    "ganesh_mlmc": "MLMC-NPG",
}
MODE_LABEL = {"ref": "ref", "moreK": "moreK", "largerN": "largerN"}
ENV_LABEL = {
    "dst": "DST 10x5",
    "dst_original": "Original DST 11x10",
    "dst_large": "Larger DST 20x10",
    "fruit_tree": "Fruit Tree",
    "bonus_world": "Bonus World",
}

RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc)_"
    r"(?P<env>.+)_"
    r"(?P<policy>tab|nn\d+(?:x\d+)*)_"
    r"N(?P<N>\d+)_K(?P<K>\d+)_"
    r"(?P<mode>ref|moreK|largerN)_"
    r"seed(?P<seed>\d+)$"
)


def plateau(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    tail = max(1, len(arr) // 10)
    return float(arr[-tail:].mean())


def load_runs(sweep_dir: Path) -> list[dict]:
    runs = []
    for fp in sorted(sweep_dir.glob("*.json")):
        if fp.name == "sweep_manifest.json":
            continue
        m = RUN_RE.match(fp.stem)
        if not m:
            print(f"skip unmatched file {fp.name}")
            continue
        with open(fp) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        run = {
            "file": str(fp),
            "algo": m["algo"],
            "env": m["env"],
            "policy": m["policy"],
            "N": int(m["N"]),
            "K": int(m["K"]),
            "mode": m["mode"],
            "seed": int(m["seed"]),
            "f": np.asarray(data["f"], dtype=float),
            "trajectories": np.asarray(data["trajectories"], dtype=float),
            "plateau": plateau(data["f"]),
            "total_trajectories": float(data["trajectories"][-1]),
            "wall_time_sec": float(data.get("wall_time_sec", 0.0)),
            "config": cfg,
        }
        runs.append(run)
    return runs


def group_key(run: dict) -> tuple:
    return (run["env"], run["policy"], run["algo"], run["N"], run["K"], run["mode"])


def summarise(runs: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for run in runs:
        by[group_key(run)].append(run)

    rows = []
    for (env, policy, algo, N, K, mode), items in sorted(by.items()):
        plateaus = np.asarray([r["plateau"] for r in items], dtype=float)
        traj = np.asarray([r["total_trajectories"] for r in items], dtype=float)
        rows.append({
            "env": env,
            "policy": policy,
            "algo": algo,
            "label": LABEL[algo],
            "N": N,
            "K": K,
            "mode": mode,
            "n_seeds": len(items),
            "plateau_mean": float(plateaus.mean()),
            "plateau_std": float(plateaus.std()),
            "traj_mean": float(traj.mean()),
            "traj_std": float(traj.std()),
        })
    return rows


def write_summary_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "env", "policy", "algo", "label", "N", "K", "mode", "n_seeds",
        "plateau_mean", "plateau_std", "traj_mean", "traj_std",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"wrote {out}")


def aggregate_curves(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for run in runs:
        by[(run["algo"], run["N"], run["K"], run["mode"])].append(run)

    out = {}
    for key, items in by.items():
        f_curves = [r["f"] for r in items]
        x_curves = [r["trajectories"] for r in items]
        L = min(len(c) for c in f_curves)
        f_stack = np.stack([c[:L] for c in f_curves])
        x_stack = np.stack([c[:L] for c in x_curves])
        plateaus = np.asarray([r["plateau"] for r in items], dtype=float)
        totals = np.asarray([r["total_trajectories"] for r in items], dtype=float)
        out[key] = {
            "f_mean": f_stack.mean(axis=0),
            "f_std": f_stack.std(axis=0),
            "x_mean": x_stack.mean(axis=0),
            "n_seeds": len(items),
            "plateau_mean": float(plateaus.mean()),
            "plateau_std": float(plateaus.std()),
            "traj_mean": float(totals.mean()),
        }
    return out


def ordered_keys(agg: dict) -> list[tuple]:
    mode_rank = {"ref": 0, "moreK": 1, "largerN": 2}
    algo_rank = {"ganesh_mlmc": 0, "bai_pg": 1, "reinforce": 2}
    return sorted(agg.keys(), key=lambda k: (algo_rank[k[0]], mode_rank[k[3]], k[1], k[2]))


def plot_case(runs: list[dict], title: str, out: Path) -> None:
    agg = aggregate_curves(runs)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]}
    )
    ax, bx = axes

    for algo, N, K, mode in ordered_keys(agg):
        stats = agg[(algo, N, K, mode)]
        color = COLOR[algo]
        ls = "-" if mode == "ref" else ("--" if mode == "moreK" else ":")
        lw = 2.1 if algo == "ganesh_mlmc" else 1.5
        label = f"{LABEL[algo]} {MODE_LABEL[mode]} N={N} K={K} (n={stats['n_seeds']})"
        ax.plot(stats["x_mean"], stats["f_mean"], color=color, ls=ls, lw=lw, label=label)
        ax.fill_between(
            stats["x_mean"],
            stats["f_mean"] - stats["f_std"],
            stats["f_mean"] + stats["f_std"],
            color=color,
            alpha=0.14 if algo == "ganesh_mlmc" else 0.08,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cumulative trajectories")
    ax.set_ylabel("f(J)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    labels, means, stds, colors = [], [], [], []
    for algo, N, K, mode in ordered_keys(agg):
        stats = agg[(algo, N, K, mode)]
        labels.append(f"{LABEL[algo].split('(')[0].strip()}\n{mode} N={N}")
        means.append(stats["plateau_mean"])
        stds.append(stats["plateau_std"])
        colors.append(COLOR[algo])

    xs = np.arange(len(labels))
    bx.bar(xs, means, yerr=stds, color=colors, alpha=0.75, capsize=3)
    bx.set_xticks(xs)
    bx.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    bx.set_ylabel("plateau f(J), per-seed last 10%")
    bx.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def row_lookup(rows: list[dict]) -> dict:
    return {(r["env"], r["policy"], r["algo"], r["mode"]): r for r in rows}


def best_row(rows: list[dict], env: str, policy: str, algos: set[str]) -> dict | None:
    candidates = [
        r for r in rows
        if r["env"] == env and r["policy"] == policy and r["algo"] in algos
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["plateau_mean"])


def plot_round_a_summary(rows: list[dict], out: Path) -> None:
    envs = ["dst", "dst_original", "dst_large", "fruit_tree", "bonus_world"]
    lookup = row_lookup(rows)
    mlmc = [lookup[(e, "tab", "ganesh_mlmc", "ref")] for e in envs if (e, "tab", "ganesh_mlmc", "ref") in lookup]

    labels = []
    margins_bai = []
    margins_any = []
    for m in mlmc:
        best_bai = best_row(rows, m["env"], "tab", {"bai_pg"})
        best_any = best_row(rows, m["env"], "tab", {"bai_pg", "reinforce"})
        if best_bai is None or best_any is None:
            continue
        labels.append(ENV_LABEL.get(m["env"], m["env"]))
        margins_bai.append(m["plateau_mean"] - best_bai["plateau_mean"])
        margins_any.append(m["plateau_mean"] - best_any["plateau_mean"])

    xs = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.bar(xs - width / 2, margins_bai, width, color="tab:purple", alpha=0.75,
           label="MLMC minus best Bai fair-budget")
    ax.bar(xs + width / 2, margins_any, width, color="tab:gray", alpha=0.65,
           label="MLMC minus best non-MLMC")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("plateau margin")
    ax.set_title("v11 Round A fair-budget margins")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fruit_tree_d(policy: str) -> int:
    # depth-7 Fruit Tree: 256 states including absorbing sentinel, 2 actions.
    if policy == "tab":
        return 256 * 2
    m = re.match(r"nn(?P<h>\d+)$", policy)
    if not m:
        raise ValueError(policy)
    h = int(m["h"])
    return 256 * h + h + h * 2 + 2


def plot_dscale(rows: list[dict], out: Path) -> None:
    policies = ["tab", "nn32", "nn128"]
    labels = ["tabular", "NN-32", "NN-128"]
    ds = np.asarray([fruit_tree_d(p) for p in policies], dtype=float)

    def get(policy: str, algo: str, mode: str) -> dict | None:
        matches = [
            r for r in rows
            if r["env"] == "fruit_tree"
            and r["policy"] == policy
            and r["algo"] == algo
            and r["mode"] == mode
        ]
        return matches[0] if matches else None

    series = [
        ("MLMC-NPG", "ganesh_mlmc", "ref", "tab:purple", "-"),
        ("Bai-PG moreK", "bai_pg", "moreK", "tab:blue", "--"),
        ("Bai-PG largerN", "bai_pg", "largerN", "tab:blue", ":"),
        (r"REINFORCE $\delta$=1 moreK", "reinforce", "moreK", "tab:red", "--"),
        (r"REINFORCE $\delta$=1 largerN", "reinforce", "largerN", "tab:red", ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax, bx = axes
    for label, algo, mode, color, ls in series:
        means, stds = [], []
        for p in policies:
            row = get(p, algo, mode)
            means.append(np.nan if row is None else row["plateau_mean"])
            stds.append(0.0 if row is None else row["plateau_std"])
        ax.errorbar(ds, means, yerr=stds, color=color, ls=ls, marker="o",
                    capsize=3, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([f"{lab}\nd={int(d)}" for lab, d in zip(labels, ds)])
    ax.set_ylabel("plateau f(J)")
    ax.set_title("Fruit Tree d-scale")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    headline, fair = [], []
    for p in policies:
        mlmc = get(p, "ganesh_mlmc", "ref")
        bai_morek = get(p, "bai_pg", "moreK")
        bai_best = best_row(rows, "fruit_tree", p, {"bai_pg"})
        if mlmc is None or bai_morek is None or bai_best is None:
            headline.append(np.nan)
            fair.append(np.nan)
        else:
            headline.append(mlmc["plateau_mean"] - bai_morek["plateau_mean"])
            fair.append(mlmc["plateau_mean"] - bai_best["plateau_mean"])

    width = 0.36
    xs = np.arange(len(policies))
    bx.axhline(0.0, color="black", lw=0.8)
    bx.bar(xs - width / 2, headline, width, color="tab:purple", alpha=0.75,
           label="MLMC minus Bai moreK")
    bx.bar(xs + width / 2, fair, width, color="tab:gray", alpha=0.65,
           label="MLMC minus best Bai fair-budget")
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{lab}\nd={int(d)}" for lab, d in zip(labels, ds)])
    bx.set_ylabel("plateau margin")
    bx.set_title("Headline vs fair-budget margin")
    bx.grid(True, axis="y", alpha=0.3)
    bx.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round_a", default="results/sweep_v11_round_a")
    p.add_argument("--round_b", default="results/sweep_v11_round_b")
    p.add_argument("--out_dir", default="figures/v11_fair_budget")
    p.add_argument("--summary_dir", default="results/v11_summaries")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    summary_dir = Path(args.summary_dir)

    round_a = load_runs(Path(args.round_a))
    round_b = load_runs(Path(args.round_b))
    rows_a = summarise(round_a)
    rows_b = summarise(round_b)
    rows_all = rows_a + rows_b

    write_summary_csv(rows_a, summary_dir / "round_a_plateaus.csv")
    write_summary_csv(rows_b, summary_dir / "round_b_plateaus.csv")
    write_summary_csv(rows_all, summary_dir / "all_plateaus.csv")

    for env in sorted({r["env"] for r in round_a}):
        runs = [r for r in round_a if r["env"] == env and r["policy"] == "tab"]
        plot_case(
            runs,
            f"{ENV_LABEL.get(env, env)} tabular fair-budget",
            out_dir / "round_a" / f"{env}_fair_budget.png",
        )

    for policy in sorted({r["policy"] for r in round_b}):
        runs = [r for r in round_b if r["env"] == "fruit_tree" and r["policy"] == policy]
        plot_case(
            runs,
            f"Fruit Tree {policy} fair-budget",
            out_dir / "round_b" / f"fruit_tree_{policy}_fair_budget.png",
        )

    plot_round_a_summary(rows_a, out_dir / "round_a_summary.png")
    plot_dscale(rows_all, out_dir / "fruit_tree_dscale.png")


if __name__ == "__main__":
    main()
