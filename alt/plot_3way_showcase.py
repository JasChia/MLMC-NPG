"""3-way showcase plot: Bai-PG vs REINFORCE (delta=1.0) vs MLMC-NPG
at q_0=0, N=4, alpha=2, M=8, on cumulative-trajectory axis.

The q_0=0 setting matches Paper A's literal statement and exposes more
curvature in ∂²f/∂J², which is what MLMC's unbiased estimator exploits.
"""
import json, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load(dirs):
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                out.append(json.load(f))
    return out


def trailing_mean(stack, w=30):
    K = stack.shape[1]
    w = min(w, K)
    cs = np.cumsum(stack, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    lo = np.maximum(0, np.arange(K) - w + 1)
    hi = np.arange(K) + 1
    counts = (hi - lo).astype(np.float64)
    return (cs[:, hi] - cs[:, lo]) / counts[None, :]


def band_by_traj(runs, n=300):
    trajs = [np.asarray(r["trajectories"]) for r in runs]
    fs = [np.asarray(r["f"]) for r in runs]
    lo = max(t[0] for t in trajs)
    hi = min(t[-1] for t in trajs)
    xs = np.linspace(lo, hi, n)
    interp = np.stack([np.interp(xs, t, f) for t, f in zip(trajs, fs)], axis=0)
    sm = trailing_mean(interp, w=30)
    return xs, sm.mean(axis=0), sm.std(axis=0)


runs = load(["results/sweep_v4_q0_0"])
groups = defaultdict(list)
for r in runs:
    groups[r["config"]["algorithm"]].append(r)

fig, ax = plt.subplots(figsize=(9.5, 5.5))

series = [
    ("bai_pg",       "tab:blue",   "-",  "Bai-PG (K=3000, N=4)"),
    ("reinforce",    "tab:red",    "-",  r"REINFORCE (K=6000, N=4, $\delta$=1.0)"),
    ("ganesh_mlmc",  "tab:purple", "--", "MLMC-NPG (K=1500, B=4, N_inner=10)"),
]

for algo, color, ls, label in series:
    rs = groups.get(algo, [])
    if not rs:
        print(f"missing: {algo}")
        continue
    x, m, s = band_by_traj(rs)
    ax.plot(x, m, color=color, linestyle=ls,
            label=f"{label}  [end traj={int(x[-1])}]", lw=1.9)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.13)
    # Print plateau
    f_last10pct = []
    for r in rs:
        f = np.asarray(r["f"])
        f_last10pct.append(f[-max(1, f.shape[0]//10):].mean())
    print(f"  {algo:<14} plateau = {np.mean(f_last10pct):>8.2f} ± {np.std(f_last10pct):>5.2f}")

# Reference: arrival-limited optimum with q_0=0
ax.axhline(-165.7, color="black", linestyle=":", linewidth=1.0,
           label="arrival-limited optimum ($q_0=0$)")

ax.set_xscale("log")
ax.set_xlabel("Cumulative trajectories (log scale)")
ax.set_ylabel(r"$f_{\alpha=2}(\hat J^{\pi_k})$")
ax.set_title(r"MLMC-NPG vs Bai-PG vs REINFORCE at $q_0=0$, N=4, M=8, $\alpha$=2")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", frameon=False, fontsize=9)

fig.tight_layout()
out = Path("figures/v4_showcase/three_way_q0_0.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nsaved {out}")
