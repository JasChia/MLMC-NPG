"""Analyze long-K (K=1000) results: did we push past the paper and the K=500 runs?"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict


def load(p):
    return [json.loads(q.read_text()) for q in sorted(Path(p).glob("*.json")) if q.name != "sweep_manifest.json"]


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        key = (c["algorithm"], c["M"], c["N"])
        groups[key].append(r["f"])
    out = {}
    for key, fs in groups.items():
        arrs = [np.asarray(f) for f in fs]
        K = min(a.shape[0] for a in arrs)
        stack = np.stack([a[:K] for a in arrs], axis=0)
        # Use the last 100 iters mean as "plateau" to smooth noise
        last_k = min(100, K)
        plateau = stack[:, -last_k:].mean(axis=1)  # per seed
        out[key] = (plateau.mean(), plateau.std(), K, len(fs))
    return out


longk = summarize(load("results/sweep_v2_longK"))
k500 = summarize(load("results/sweep_v2_loglinear"))

print(f"{'algo':<12} {'M':>3} {'N':>5} | {'K=500 (v2)':>16} | {'K=1000':>16} | {'Δ':>8}")
print("-" * 75)
for key in sorted(longk):
    algo, M, N = key
    lm, ls, lK, ln = longk[key]
    km, ks, kK, kn = k500.get(key, (float('nan'), 0, 0, 0))
    delta = lm - km
    print(f"{algo:<12} {M:>3} {N:>5} | "
          f"{km:>8.3f} ± {ks:>4.2f} ({kn}s) | "
          f"{lm:>8.3f} ± {ls:>4.2f} ({ln}s) | "
          f"{delta:>+7.3f}")
