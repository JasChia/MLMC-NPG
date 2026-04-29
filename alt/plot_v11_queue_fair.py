"""Plots and tables for v11 queueing fair-budget runs."""
from __future__ import annotations

import argparse
import csv
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
RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc)_log_linear_"
    r"M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)_"
    r"(?P<mode>ref|moreK|largerN)_alpha(?P<alpha>[\d.]+)_seed(?P<seed>\d+)$"
)


def plateau(f) -> float:
    arr = np.asarray(f, dtype=float)
    tail = max(1, len(arr) // 10)
    return float(arr[-tail:].mean())


def load_runs(sweep_dir: Path) -> list[dict]:
    out = []
    for fp in sorted(sweep_dir.glob("*.json")):
        if fp.name == "sweep_manifest.json":
            continue
        m = RUN_RE.match(fp.stem)
        if not m:
            print(f"skip unmatched file {fp.name}")
            continue
        with open(fp) as f:
            d = json.load(f)
        out.append({
            "file": str(fp),
            "algo": m["algo"],
            "M": int(m["M"]),
            "N": int(m["N"]),
            "K": int(m["K"]),
            "mode": m["mode"],
            "alpha": float(m["alpha"]),
            "seed": int(m["seed"]),
            "f": np.asarray(d["f"], dtype=float),
            "trajectories": np.asarray(d["trajectories"], dtype=float),
            "plateau": plateau(d["f"]),
            "traj_total": float(d["trajectories"][-1]),
        })
    return out


def summarise(runs: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in runs:
        by[(r["M"], r["algo"], r["N"], r["K"], r["mode"])].append(r)
    rows = []
    for (M, algo, N, K, mode), items in sorted(by.items()):
        p = np.asarray([r["plateau"] for r in items], dtype=float)
        t = np.asarray([r["traj_total"] for r in items], dtype=float)
        rows.append({
            "M": M,
            "algo": algo,
            "label": LABEL[algo],
            "N": N,
            "K": K,
            "mode": mode,
            "n_seeds": len(items),
            "plateau_mean": float(p.mean()),
            "plateau_std": float(p.std()),
            "traj_mean": float(t.mean()),
            "traj_std": float(t.std()),
        })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "M", "algo", "label", "N", "K", "mode", "n_seeds",
        "plateau_mean", "plateau_std", "traj_mean", "traj_std",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"wrote {out}")


def aggregate(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[(r["algo"], r["N"], r["K"], r["mode"])].append(r)
    out = {}
    for key, items in by.items():
        L = min(len(r["f"]) for r in items)
        f = np.stack([r["f"][:L] for r in items])
        x = np.stack([r["trajectories"][:L] for r in items])
        p = np.asarray([r["plateau"] for r in items])
        out[key] = {
            "f_mean": f.mean(axis=0),
            "f_std": f.std(axis=0),
            "x_mean": x.mean(axis=0),
            "plateau_mean": float(p.mean()),
            "plateau_std": float(p.std()),
            "n_seeds": len(items),
        }
    return out


def ordered_keys(agg: dict) -> list[tuple]:
    algo_rank = {"ganesh_mlmc": 0, "bai_pg": 1, "reinforce": 2}
    mode_rank = {"ref": 0, "moreK": 1, "largerN": 2}
    return sorted(agg.keys(), key=lambda k: (algo_rank[k[0]], mode_rank[k[3]], k[1], k[2]))


def plot_case(runs: list[dict], M: int, out: Path) -> None:
    agg = aggregate(runs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    ax, bx = axes
    for algo, N, K, mode in ordered_keys(agg):
        s = agg[(algo, N, K, mode)]
        color = COLOR[algo]
        ls = "-" if mode == "ref" else ("--" if mode == "moreK" else ":")
        label = f"{LABEL[algo]} {mode} N={N} K={K} (n={s['n_seeds']})"
        ax.plot(s["x_mean"], s["f_mean"], color=color, ls=ls, lw=2 if algo == "ganesh_mlmc" else 1.5, label=label)
        ax.fill_between(s["x_mean"], s["f_mean"] - s["f_std"], s["f_mean"] + s["f_std"],
                        color=color, alpha=0.14)
    ax.set_title(f"Queueing M={M}, log-linear, alpha=2 fair budget")
    ax.set_xlabel("cumulative trajectories")
    ax.set_ylabel("f(J)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    labels, means, stds, colors = [], [], [], []
    for algo, N, _, mode in ordered_keys(agg):
        s = agg[(algo, N, _, mode)]
        labels.append(f"{LABEL[algo].split('(')[0].strip()}\n{mode} N={N}")
        means.append(s["plateau_mean"])
        stds.append(s["plateau_std"])
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


def plot_summary(rows: list[dict], out: Path) -> None:
    Ms = sorted({r["M"] for r in rows})
    margins_bai, margins_any = [], []
    labels = [f"M={M}" for M in Ms]
    for M in Ms:
        mlmc = [r for r in rows if r["M"] == M and r["algo"] == "ganesh_mlmc"][0]
        best_bai = max([r for r in rows if r["M"] == M and r["algo"] == "bai_pg"],
                       key=lambda r: r["plateau_mean"])
        best_any = max([r for r in rows if r["M"] == M and r["algo"] != "ganesh_mlmc"],
                       key=lambda r: r["plateau_mean"])
        margins_bai.append(mlmc["plateau_mean"] - best_bai["plateau_mean"])
        margins_any.append(mlmc["plateau_mean"] - best_any["plateau_mean"])
    xs = np.arange(len(Ms))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.bar(xs - width / 2, margins_bai, width, color="tab:purple", alpha=0.75,
           label="MLMC minus best Bai fair-budget")
    ax.bar(xs + width / 2, margins_any, width, color="tab:gray", alpha=0.65,
           label="MLMC minus best non-MLMC")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("plateau margin")
    ax.set_title("v11 queueing fair-budget margins")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="results/sweep_v11_queue_fair")
    p.add_argument("--out_dir", default="figures/v11_queue_fair")
    p.add_argument("--summary_csv", default="results/v11_summaries/queue_fair_plateaus.csv")
    args = p.parse_args()

    runs = load_runs(Path(args.sweep_dir))
    rows = summarise(runs)
    write_csv(rows, Path(args.summary_csv))
    out_dir = Path(args.out_dir)
    for M in sorted({r["M"] for r in runs}):
        plot_case([r for r in runs if r["M"] == M], M, out_dir / f"queue_M{M}_fair_budget.png")
    plot_summary(rows, out_dir / "queue_fair_summary.png")


if __name__ == "__main__":
    main()
