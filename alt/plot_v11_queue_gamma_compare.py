"""Compare queue fair-budget results at gamma=0.9999 vs gamma=0.99."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path, gamma_label: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["gamma"] = gamma_label
            row["M"] = int(row["M"])
            row["N"] = int(row["N"])
            row["K"] = int(row["K"])
            row["plateau_mean"] = float(row["plateau_mean"])
            row["plateau_std"] = float(row["plateau_std"])
            rows.append(row)
    return rows


def best_non_mlmc(rows: list[dict], M: int) -> dict:
    return max(
        [r for r in rows if r["M"] == M and r["algo"] != "ganesh_mlmc"],
        key=lambda r: r["plateau_mean"],
    )


def mlmc_row(rows: list[dict], M: int) -> dict:
    matches = [r for r in rows if r["M"] == M and r["algo"] == "ganesh_mlmc"]
    if len(matches) != 1:
        raise ValueError((M, len(matches)))
    return matches[0]


def write_compare(rows_by_gamma: dict[str, list[dict]], out: Path) -> list[dict]:
    compare = []
    for gamma, rows in rows_by_gamma.items():
        for M in sorted({r["M"] for r in rows}):
            m = mlmc_row(rows, M)
            b = best_non_mlmc(rows, M)
            compare.append({
                "gamma": gamma,
                "M": M,
                "mlmc_mean": m["plateau_mean"],
                "mlmc_std": m["plateau_std"],
                "best_baseline": f"{b['algo']} {b['mode']} N={b['N']}",
                "best_baseline_mean": b["plateau_mean"],
                "best_baseline_std": b["plateau_std"],
                "mlmc_minus_best": m["plateau_mean"] - b["plateau_mean"],
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gamma", "M", "mlmc_mean", "mlmc_std", "best_baseline",
        "best_baseline_mean", "best_baseline_std", "mlmc_minus_best",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in compare:
            w.writerow(row)
    print(f"wrote {out}")
    return compare


def plot_compare(compare: list[dict], out: Path) -> None:
    Ms = sorted({r["M"] for r in compare})
    gammas = ["0.9999", "0.99"]
    xs = np.arange(len(Ms))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="black", lw=0.8)
    for i, gamma in enumerate(gammas):
        vals = [
            next(r for r in compare if r["gamma"] == gamma and r["M"] == M)["mlmc_minus_best"]
            for M in Ms
        ]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f"gamma={gamma}", alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"M={M}" for M in Ms])
    ax.set_ylabel("MLMC plateau minus best non-MLMC")
    ax.set_title("Queueing gamma sensitivity, fair-budget margin")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper_csv", default="results/v11_summaries/queue_fair_plateaus.csv")
    p.add_argument("--gamma099_csv", default="results/v11_summaries/queue_fair_gamma099_plateaus.csv")
    p.add_argument("--out_csv", default="results/v11_summaries/queue_gamma_compare.csv")
    p.add_argument("--out_fig", default="figures/v11_queue_gamma_compare/queue_gamma_compare.png")
    args = p.parse_args()

    rows_by_gamma = {
        "0.9999": load_rows(Path(args.paper_csv), "0.9999"),
        "0.99": load_rows(Path(args.gamma099_csv), "0.99"),
    }
    compare = write_compare(rows_by_gamma, Path(args.out_csv))
    plot_compare(compare, Path(args.out_fig))


if __name__ == "__main__":
    main()
