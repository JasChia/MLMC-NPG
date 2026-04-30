"""Round analyzer for v17 search.

For each tag in a results dir, computes mean final-f per algorithm across
seeds and reports MLMC's lead vs Bai. Sorts by MLMC lead (desc).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_RE = re.compile(
    r"(?P<algo>bai_pg|reinforce|ganesh_mlmc|scalarized_reinforce)_"
    r"(?P<env_tag>.+?)_"
    r"(?P<policy>tab|nn\d+(?:x\d+)*(?:_[a-z_]+)?)_"
    r"B(?P<B>\d+)_N(?P<N>\d+)_K(?P<K>\d+)_"
    r"beta(?P<beta>[^_]+)_lr(?P<lr>[^_]+)(?:_o[a-z_]+)?_"
    r"v(?P<version>1[45])_(?P<tag>.+)_seed(?P<seed>\d+)"
)


def load_runs(results_dir: Path):
    by_tag = defaultdict(lambda: defaultdict(list))   # tag -> algo -> [final_f]
    by_tag_traj = defaultdict(lambda: defaultdict(list))  # tag -> algo -> [traj_count]
    for fp in sorted(results_dir.glob("*.json")):
        if "manifest" in fp.name:
            continue
        m = RUN_RE.match(fp.stem)
        if not m:
            continue
        with open(fp) as f:
            d = json.load(f)
        f_arr = np.asarray(d["f"], dtype=float)
        tail = max(1, len(f_arr) // 10)
        plateau = float(f_arr[-tail:].mean())
        traj = d["trajectories"][-1]
        by_tag[m.group("tag")][m.group("algo")].append(plateau)
        by_tag_traj[m.group("tag")][m.group("algo")].append(traj)
    return by_tag, by_tag_traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    by_tag, by_tag_traj = load_runs(Path(args.results_dir))
    rows = []
    for tag, algos in by_tag.items():
        n = min((len(v) for v in algos.values()), default=0)
        if n == 0:
            continue
        means = {a: float(np.mean(v)) for a, v in algos.items()}
        sds = {a: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for a, v in algos.items()}
        mlmc = means.get("ganesh_mlmc", float("nan"))
        bai = means.get("bai_pg", float("nan"))
        rein = means.get("reinforce", float("nan"))
        scal = means.get("scalarized_reinforce", float("nan"))
        traj_mlmc = float(np.mean(by_tag_traj[tag].get("ganesh_mlmc", [0])))
        traj_bai = float(np.mean(by_tag_traj[tag].get("bai_pg", [0])))
        rows.append({
            "tag": tag,
            "n_seeds": n,
            "MLMC": mlmc, "MLMC_sd": sds.get("ganesh_mlmc", 0.0),
            "Bai": bai, "Bai_sd": sds.get("bai_pg", 0.0),
            "REINF": rein,
            "scalREINF": scal,
            "M-Bai": mlmc - bai if not (np.isnan(mlmc) or np.isnan(bai)) else float("nan"),
            "M-REINF": mlmc - rein if not (np.isnan(mlmc) or np.isnan(rein)) else float("nan"),
            "M-scal": mlmc - scal if not (np.isnan(mlmc) or np.isnan(scal)) else float("nan"),
            "traj_M": traj_mlmc, "traj_B": traj_bai,
        })
    rows.sort(key=lambda r: -r["M-Bai"] if not np.isnan(r["M-Bai"]) else -1e9)

    print(f"=== v17 round analysis: {args.results_dir} ({len(rows)} configs) ===")
    print(f"{'tag':<55} {'n':>2} {'MLMC':>7} {'Bai':>7} {'REINF':>7} {'scal':>7} {'M-Bai':>7} {'M-REINF':>8} {'M-scal':>7}")
    print("-" * 120)
    for r in rows[:args.top]:
        print(f"{r['tag']:<55} {r['n_seeds']:>2} "
              f"{r['MLMC']:>7.2f} {r['Bai']:>7.2f} {r['REINF']:>7.2f} {r['scalREINF']:>7.2f} "
              f"{r['M-Bai']:>+7.2f} {r['M-REINF']:>+8.2f} {r['M-scal']:>+7.2f}")
    print("\n... (worst 5)")
    for r in rows[-5:]:
        print(f"{r['tag']:<55} {r['n_seeds']:>2} "
              f"{r['MLMC']:>7.2f} {r['Bai']:>7.2f} {r['REINF']:>7.2f} {r['scalREINF']:>7.2f} "
              f"{r['M-Bai']:>+7.2f} {r['M-REINF']:>+8.2f} {r['M-scal']:>+7.2f}")


if __name__ == "__main__":
    main()
