"""Summarize MLMC-showcase sweep: plateau f-value per (algo, alpha), by seed."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict


runs = []
for p in sorted(Path("results/sweep_v3_mlmc").glob("*.json")):
    if p.name == "sweep_manifest.json":
        continue
    with open(p) as f:
        runs.append(json.load(f))

groups = defaultdict(list)
for r in runs:
    c = r["config"]
    key = (c["algorithm"], c["alpha"])
    groups[key].append(r)

print(f"{'algo':<14} {'alpha':>6} {'f_final_last100':>20} "
      f"{'f_mid_last100@K=250':>22} {'traj_used':>12}")
print("-" * 80)
for (algo, alpha), rs in sorted(groups.items()):
    f_finals, f_mids, trajs = [], [], []
    for r in rs:
        f = np.asarray(r["f"])
        K = f.shape[0]
        # Last 100 iters mean as "final"
        f_finals.append(f[-100:].mean())
        # Take iters around K/2 (fair at mid-training)
        mid = K // 2
        f_mids.append(f[max(0, mid - 50): mid + 50].mean())
        trajs.append(r["trajectories"][-1])
    f_finals = np.array(f_finals)
    f_mids = np.array(f_mids)
    trajs = np.array(trajs)
    print(f"{algo:<14} {alpha:>6.1f} "
          f"{f_finals.mean():>10.3f} ± {f_finals.std():>5.2f}    "
          f"{f_mids.mean():>10.3f} ± {f_mids.std():>5.2f}     "
          f"{trajs.mean():>10.0f}")
