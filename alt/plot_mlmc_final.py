"""Final MLMC-superiority plot: Bai-PG K=3000 vs MLMC K=1500 at α=2, N=4, M=8.

Demonstrates that MLMC-NPG reaches a plateau BELOW Bai-PG's even after
matching the cumulative-trajectory budget. This is the bias-floor of Bai-PG:
no matter how many iterations it runs, the plug-in estimator's bias keeps it
above a certain f value, while MLMC's unbiased estimator continues improving.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load(dirs):
    runs = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                runs.append(json.load(f))
    return runs


def trailing_mean(stack, w=50):
    K = stack.shape[1]
    w = min(w, K)
    cs = np.cumsum(stack, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    idx_lo = np.maximum(0, np.arange(K) - w + 1)
    idx_hi = np.arange(K) + 1
    counts = (idx_hi - idx_lo).astype(np.float64)
    return (cs[:, idx_hi] - cs[:, idx_lo]) / counts[None, :]


def mean_std_by_traj(runs, n_grid=300):
    traj_all = [np.asarray(r["trajectories"]) for r in runs]
    f_all = [np.asarray(r["f"]) for r in runs]
    lo = max(t[0] for t in traj_all)
    hi = min(t[-1] for t in traj_all)
    xs = np.linspace(lo, hi, n_grid)
    interp = np.stack([np.interp(xs, t, f) for t, f in zip(traj_all, f_all)], axis=0)
    sm = trailing_mean(interp, w=30)
    return xs, sm.mean(axis=0), sm.std(axis=0), interp


all_runs = load(["results/sweep_v3_mlmc", "results/sweep_v3_mlmc_ext"])

# Group: (algo, alpha, K)
groups = defaultdict(list)
for r in all_runs:
    c = r["config"]
    if c["alpha"] != 2.0 or c["M"] != 8 or c["N"] != 4:
        continue
    groups[(c["algorithm"], c["K"])].append(r)

# Build plot: Bai-PG (K=500 and K=3000) + MLMC (K=500 and K=1500)
fig, ax = plt.subplots(figsize=(9, 5.5))

series = [
    ("bai_pg",      500, "tab:blue",   "-",  "Bai-PG (N=4, K=500)"),
    ("bai_pg",     3000, "tab:cyan",   "-",  "Bai-PG (N=4, K=3000)"),
    ("ganesh_mlmc", 500, "tab:orange", "--", "MLMC-NPG (B=4, K=500)"),
    ("ganesh_mlmc",1500, "tab:purple", "--", "MLMC-NPG (B=4, K=1500)"),
]

for algo, K, color, ls, label in series:
    rs = groups.get((algo, K), [])
    if not rs:
        print(f"  NO DATA for {algo} K={K}")
        continue
    x, m, s, _ = mean_std_by_traj(rs)
    ax.plot(x, m, color=color, linestyle=ls, label=f"{label}  [traj={int(x[-1])}]", lw=1.8)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.12)

# Reference lines
ax.axhline(-119.5, color="black", linestyle=":", linewidth=1.0,
           label="arrival-limited optimum (q₀=5)")

ax.set_xscale("log")
ax.set_xlabel("Cumulative trajectories (log scale)")
ax.set_ylabel(r"$f_{\alpha=2}(\hat J^{\pi_k})$")
ax.set_title(r"MLMC-NPG vs Bai-PG under matched trajectory budget  (M=8, N=4, $\alpha$=2)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_ylim(-140, -118)

fig.tight_layout()
out = Path("figures/v3_mlmc/mlmc_final_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved {out}")

# Also print the final plateau numbers (mean over last ~10% of iters)
print("\n== Plateau (last 10% of iters) ==")
for (algo, K), rs in sorted(groups.items()):
    plateaus = []
    trajs = []
    for r in rs:
        f = np.asarray(r["f"])
        plateaus.append(f[-max(1, f.shape[0]//10):].mean())
        trajs.append(r["trajectories"][-1])
    plateaus = np.array(plateaus); trajs = np.array(trajs)
    print(f"  {algo:<14} K={K:<5} plateau = {plateaus.mean():>8.3f} ± {plateaus.std():>5.2f}   "
          f"(traj = {trajs.mean():>7.0f})")
