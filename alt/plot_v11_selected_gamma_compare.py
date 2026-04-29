"""Compare selected v11 environments at gamma=0.99 vs gamma=0.9999."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = [
    ("bonus_world", "tab", "Bonus World"),
    ("resource", "tab", "Resource Gathering"),
    ("fruit_tree", "nn32", "Fruit Tree NN-32"),
]


def load_rows(path: Path, gamma: str) -> list[dict]:
    rows = []
    seen = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["env"], row["policy"], row["algo"], row["N"],
                row["K"], row["mode"], gamma,
            )
            if key in seen:
                continue
            seen.add(key)
            row = dict(row)
            row["gamma"] = gamma
            row["N"] = int(row["N"])
            row["K"] = int(row["K"])
            row["plateau_mean"] = float(row["plateau_mean"])
            row["plateau_std"] = float(row["plateau_std"])
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for env, policy, label in CASES:
        for gamma in sorted({r["gamma"] for r in rows}, reverse=True):
            subset = [r for r in rows if r["env"] == env and r["policy"] == policy and r["gamma"] == gamma]
            if not subset:
                continue
            mlmc = [r for r in subset if r["algo"] == "ganesh_mlmc"][0]
            best = max([r for r in subset if r["algo"] != "ganesh_mlmc"], key=lambda r: r["plateau_mean"])
            out.append({
                "case": label,
                "env": env,
                "policy": policy,
                "gamma": gamma,
                "mlmc_mean": mlmc["plateau_mean"],
                "mlmc_std": mlmc["plateau_std"],
                "best_baseline": f"{best['algo']} {best['mode']} N={best['N']}",
                "best_baseline_mean": best["plateau_mean"],
                "best_baseline_std": best["plateau_std"],
                "mlmc_minus_best": mlmc["plateau_mean"] - best["plateau_mean"],
            })
    return out


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "env", "policy", "gamma", "mlmc_mean", "mlmc_std",
        "best_baseline", "best_baseline_mean", "best_baseline_std",
        "mlmc_minus_best",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out}")


def plot(rows: list[dict], out: Path) -> None:
    labels = [c[2] for c in CASES]
    gammas = ["0.99", "0.9999"]
    xs = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0.0, color="black", lw=0.8)
    for i, gamma in enumerate(gammas):
        vals = []
        for label in labels:
            match = [r for r in rows if r["case"] == label and r["gamma"] == gamma]
            vals.append(np.nan if not match else match[0]["mlmc_minus_best"])
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f"gamma={gamma}", alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("MLMC plateau minus best non-MLMC")
    ax.set_title("Selected v11 environments: gamma sensitivity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gamma099_csv", default="results/v11_summaries/all_plateaus.csv")
    p.add_argument("--gamma9999_csv", default="results/v11_summaries/gamma9999_selected/all_plateaus.csv")
    p.add_argument("--out_csv", default="results/v11_summaries/selected_gamma_compare.csv")
    p.add_argument("--out_fig", default="figures/v11_selected_gamma_compare/selected_gamma_compare.png")
    args = p.parse_args()

    rows = load_rows(Path(args.gamma099_csv), "0.99") + load_rows(Path(args.gamma9999_csv), "0.9999")
    summary = summarize(rows)
    write_csv(summary, Path(args.out_csv))
    plot(summary, Path(args.out_fig))


if __name__ == "__main__":
    main()
