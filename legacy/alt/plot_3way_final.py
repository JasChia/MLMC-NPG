"""Final 3-way showcase: Bai-PG vs REINFORCE (delta=1.0, paper-faithful) vs MLMC-NPG
at q_0=5, N=4, alpha=2, M=8. Matched trajectory budget (~24k for Bai-PG/REINFORCE
via K scaling, ~71k for MLMC at K=1500 to show asymptotic regime).
"""
import json, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load(dirs, filt=None):
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "sweep_manifest.json":
                continue
            with open(p) as f:
                r = json.load(f)
            if filt is None or filt(r):
                out.append(r)
    return out


def trailing_mean(stack, w=30):
    K = stack.shape[1]; w = min(w, K)
    cs = np.cumsum(stack, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    lo = np.maximum(0, np.arange(K) - w + 1); hi = np.arange(K) + 1
    counts = (hi - lo).astype(np.float64)
    return (cs[:, hi] - cs[:, lo]) / counts[None, :]


def band_by_traj(runs, n=300):
    trajs = [np.asarray(r["trajectories"]) for r in runs]
    fs = [np.asarray(r["f"]) for r in runs]
    lo = max(t[0] for t in trajs); hi = min(t[-1] for t in trajs)
    xs = np.linspace(lo, hi, n)
    interp = np.stack([np.interp(xs, t, f) for t, f in zip(trajs, fs)], axis=0)
    sm = trailing_mean(interp, w=30)
    return xs, sm.mean(axis=0), sm.std(axis=0)


def want(algo, alpha=2.0, M=8, N=4, q_init=5):
    def f(r):
        c = r["config"]
        return (c["algorithm"] == algo and c["alpha"] == alpha
                and c["M"] == M and c["N"] == N
                and (c.get("q_init") == q_init or c.get("q_init") == str(q_init)))
    return f


bai_runs = load(["results/sweep_v3_mlmc_ext"], filt=want("bai_pg"))
mlmc_runs = load(["results/sweep_v3_mlmc_ext"], filt=want("ganesh_mlmc"))
rf_runs = load(["results/sweep_v4_q0_5_rf_delta1"], filt=want("reinforce"))

print(f"Bai-PG runs: {len(bai_runs)}  K={bai_runs[0]['config']['K'] if bai_runs else '-'}")
print(f"MLMC runs:   {len(mlmc_runs)} K={mlmc_runs[0]['config']['K'] if mlmc_runs else '-'}")
print(f"RF runs:     {len(rf_runs)}  K={rf_runs[0]['config']['K'] if rf_runs else '-'}")

if not (bai_runs and mlmc_runs and rf_runs):
    raise SystemExit("missing data")

fig, ax = plt.subplots(figsize=(10, 5.5))

series = [
    (bai_runs,  "tab:blue",   "-",
     f"Bai-PG (K={bai_runs[0]['config']['K']}, N=4)"),
    (rf_runs,   "tab:red",    "-",
     rf"REINFORCE (K={rf_runs[0]['config']['K']}, N=4, $\delta$=1.0 paper-faithful)"),
    (mlmc_runs, "tab:purple", "--",
     f"MLMC-NPG (K={mlmc_runs[0]['config']['K']}, B=4, N_inner=10)"),
]

for rs, color, ls, label in series:
    x, m, s = band_by_traj(rs)
    ax.plot(x, m, color=color, linestyle=ls,
            label=f"{label}  [end traj={int(x[-1])}]", lw=1.9)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.13)
    f_last = [np.asarray(r["f"])[-max(1, len(r["f"])//10):].mean() for r in rs]
    print(f"  plateau ({label[:15]}): {np.mean(f_last):>8.2f} ± {np.std(f_last):>5.2f} "
          f"(n={len(rs)})")

ax.axhline(-119.5, color="black", linestyle=":", linewidth=1.0,
           label=r"arrival-limited optimum ($q_0=5$)")

ax.set_xscale("log")
ax.set_xlabel("Cumulative trajectories (log scale, fair per-sample comparison)")
ax.set_ylabel(r"$f_{\alpha=2}(\hat J^{\pi_k})$")
ax.set_title("3-way showcase: MLMC-NPG vs Bai-PG vs paper-faithful REINFORCE  "
             r"($q_0=5$, N=4, M=8, $\alpha$=2)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_ylim(-140, -118)

fig.tight_layout()
out = Path("figures/v4_showcase/three_way_final.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nsaved {out}")
