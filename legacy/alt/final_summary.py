"""Final summary table aggregating all v2 sweeps + paper + v1 baselines."""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def load(path):
    return [json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json")) if p.name != "sweep_manifest.json"]


def plateau_stats(runs, last_k=100):
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        key = (c["algorithm"], c.get("policy", "log_linear"), c["M"], c["N"])
        g[key].append(np.asarray(r["f"]))
    out = {}
    for k, arrs in g.items():
        K = min(a.shape[0] for a in arrs)
        stack = np.stack([a[:K] for a in arrs], axis=0)
        last = stack[:, -min(last_k, K):].mean(axis=1)
        out[k] = (float(last.mean()), float(last.std()), K, len(arrs))
    return out


sweeps = {
    "v1_loglinear": "results/sweep_fig1_fig2",
    "v2_loglinear": "results/sweep_v2_loglinear",
    "v2_tabular":   "results/sweep_v2_tabular",
    "v2_nn":        "results/sweep_v2_nn",
    "v2_longK":     "results/sweep_v2_longK",
}

stats = {name: plateau_stats(load(path)) for name, path in sweeps.items()}

# Build a focused comparison table at N=256.
PAPER = {("bai_pg", "log_linear", 2, 256): -7.60,
         ("bai_pg", "log_linear", 4, 256): -25.30,
         ("bai_pg", "log_linear", 8, 256): -126.0,
         ("reinforce", "log_linear", 2, 256): -7.75,
         ("reinforce", "log_linear", 4, 256): -26.00,
         ("reinforce", "log_linear", 8, 256): -128.0}

print("\n== Paper reproduction at N=256 (log_linear softmax) ==\n")
print(f"{'algo':<12} {'M':>3} | {'paper':>8} | "
      f"{'v1 K=500 (q0=0)':>18} | {'v2 K=500 (q0=5)':>18} | "
      f"{'v2 K=1000':>16}")
print("-" * 95)
for (algo, pol, M, N), target in sorted(PAPER.items()):
    if pol != "log_linear":
        continue
    v1 = stats["v1_loglinear"].get((algo, pol, M, N), None)
    v2 = stats["v2_loglinear"].get((algo, pol, M, N), None)
    longK = stats["v2_longK"].get((algo, pol, M, N), None)
    def fmt(x):
        return f"{x[0]:>6.2f} ± {x[1]:>4.2f} (K={x[2]})" if x else "  —"
    print(f"{algo:<12} {M:>3} | {target:>8.2f} | "
          f"{fmt(v1):>18} | {fmt(v2):>18} | "
          f"{fmt(longK):>16}")

print("\n== Tabular M=2 at N=256 (v2, K=300) ==\n")
for algo in ("bai_pg", "reinforce", "ganesh_npg", "ganesh_mlmc"):
    x = stats["v2_tabular"].get((algo, "tabular", 2, 256))
    if x:
        print(f"  {algo:<15} {x[0]:>7.3f} ± {x[1]:>4.2f} (K={x[2]}, seeds={x[3]})")

print("\n== NN at N=64 (v2, K=300 CPU / K=100 CUDA) ==\n")
for M in (2, 4, 8):
    for algo in ("bai_pg", "reinforce", "ganesh_npg", "ganesh_mlmc"):
        x = stats["v2_nn"].get((algo, "nn", M, 64))
        if x:
            print(f"  M={M} {algo:<15} {x[0]:>8.2f} ± {x[1]:>5.2f} (K={x[2]})")
