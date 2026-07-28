"""Sampling cost vs. gradient accuracy for the two return estimators.

Measures the scalarization-gradient bias

    b  =  || E[ d f(estimate) ] - d f(J^pi) ||_inf

of the two estimators the paper's algorithms differ in -- the empirical
plug-in estimator d f(Jhat_{H,B}) of Eq. (13) (Vanilla NPG, Algorithm 1) and
the truncated MLMC estimator of Eq. (14) (MLMC-NPG, Algorithm 2) -- at a
fixed policy on Fruit Tree, together with each estimator's sampling cost
(trajectories consumed per gradient estimate: B for the plug-in, the
measured E[B_k] = O(log2 B_max) for MLMC).  Plotting cost against 1/b gives
the per-gradient sample-efficiency curves: under a utility whose gradient is
Lipschitz but not second-order smooth the plug-in requires B = Theta(eps^-2)
trajectories to reach gradient accuracy eps (Lemmas 3-4), while MLMC reaches
the same accuracy at logarithmic expected cost (Lemma 5).

Utility: f(x) = sum_m [ x_m - 1/2 ((x_m - c_m)_+)^2 ], with the per-channel
kink placed at the evaluation policy's return, c = J^pi.  Its gradient
1 - (x_m - c_m)_+ is 1-Lipschitz (Assumption 2 holds with L_f = 1) but not
differentiable at the kink (Assumption 6 fails); with the return on the kink
the plug-in bias equals sigma_m / sqrt(2 pi B) in the CLT limit -- the worst
case the O(B^{-1/2}) analysis concerns.

Fruit Tree admits exact evaluation: transitions are deterministic given
actions and an episode is `depth` steps ending in one of 2^depth leaves, so
for any policy the leaf-path probabilities -- hence J^pi, the per-channel
return noise sigma, and the reference d f(J^pi) -- are enumerated in closed
form.  Leaf rewards are divided by 10 so r_m in [0, 1] and returns lie in
[0, 1]^M.  The evaluation policy is a tabular softmax over internal nodes
with logits initialized to zero (the uniform policy) plus a fixed
0.7 * N(0, I) perturbation, then held fixed.

Each configuration's bias is the average of `--reps` independent estimator
replications (each replication itself consumes B, resp. B_k, trajectories);
per-point Monte-Carlo standard errors (of the channel attaining the max) are
recorded.  For speed the replications are drawn in a vectorized,
distributionally identical form: a batch-B return mean equals counts @ R / B
with counts ~ Multinomial(B, p), and the MLMC nested coupling is preserved
by drawing the first trajectory explicitly, the rest of the first half as
Multinomial(B_k/2 - 1, p), and the second half as Multinomial(B_k/2, p).

Run:
    cd key_plots/scripts
    python samples_to_eps_lipschitz.py                # -> ../samples_to_eps_lipschitz.csv
    python samples_to_eps_lipschitz.py --reps 100000  # quick, noisier
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Repo root = parent of "key_plots"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.envs.fruit_tree import FruitTreeEnv, M_REWARD

DEPTH = 7
GAMMA = 0.99
REWARD_SCALE = 10.0     # leaf rewards are in [3, 10]; divide -> r_m in [0, 1]
POLICY_SEED = 0
DATA_SEED = 20260727
B_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CHUNK = 250_000

torch.set_default_dtype(torch.float64)


class LipschitzOnlyUtility:
    """f(x) = sum_m [x_m - 1/2 ((x_m - c_m)_+)^2];  d_m f = 1 - (x_m - c_m)_+.

    Concave and increasing on [0, 1]^M for c >= 0; the gradient is
    1-Lipschitz (Assumption 2, L_f = 1) with a jump in the second derivative
    at x_m = c_m (Assumption 6 fails).  `grad` broadcasts over (n, M) arrays.
    """

    def __init__(self, c):
        self.c = np.asarray(c, float)

    def grad(self, J):
        return 1.0 - np.maximum(0.0, np.asarray(J, float) - self.c)


def frozen_policy_leaf_probs(depth: int, seed: int) -> np.ndarray:
    """Exact leaf distribution of the frozen evaluation policy.

    Tabular softmax over the 2^depth - 1 internal nodes, logits zero (uniform
    policy) + 0.7 * N(0, I) with a fixed torch generator; the probability of
    leaf l is the product of the softmax action probabilities on its path.
    """
    n_internal = (1 << depth) - 1
    theta = torch.zeros(n_internal, 2)
    g = torch.Generator().manual_seed(seed)
    theta += 0.7 * torch.randn(theta.shape, generator=g)

    nodes, acts = [], []
    for leaf in range(1 << depth):
        node, ns, as_ = 0, [], []
        for b in range(depth):
            bit = (leaf >> (depth - 1 - b)) & 1
            ns.append(node)
            as_.append(bit)
            node = 2 * node + 1 + bit
        nodes.append(ns)
        acts.append(as_)
    logp = torch.log_softmax(theta, dim=1)
    step = logp[torch.tensor(nodes), torch.tensor(acts)]      # (2^d, depth)
    p = torch.exp(step.sum(dim=1)).numpy()
    return p / p.sum()


def plugin_stats(obj, p, R, J_true, B, reps, rng):
    """(max-channel bias, SE of that channel) of the Eq. (13) estimator."""
    s1 = np.zeros(M_REWARD)
    s2 = np.zeros(M_REWARD)
    done = 0
    while done < reps:
        r = min(CHUNK, reps - done)
        counts = rng.multinomial(B, p, size=r)
        g = obj.grad((counts @ R) / B)
        s1 += g.sum(axis=0)
        s2 += (g * g).sum(axis=0)
        done += r
    mean = s1 / reps
    var = np.maximum(s2 / reps - mean ** 2, 0.0)
    dev = np.abs(mean - obj.grad(J_true))
    m = int(np.argmax(dev))
    return float(dev[m]), float(np.sqrt(var[m] / reps))


def mlmc_stats(obj, p, R, J_true, B_max, reps, rng):
    """(max-channel bias, SE, mean trajectory cost) of the Eq. (14) estimator.

    Level Q ~ Geom(1/2); levels with 2^Q > B_max are truncated to the
    single-trajectory term.  Nested coupling: Jhat_{2^{Q-1}} uses the first
    half of the 2^Q trajectories that form Jhat_{2^Q}.
    """
    Q_max = int(np.floor(np.log2(B_max)))
    s1 = np.zeros(M_REWARD)
    s2 = np.zeros(M_REWARD)
    cost = 0.0
    done = 0
    while done < reps:
        r = min(CHUNK, reps - done)
        Q = rng.geometric(0.5, size=r)
        for q in np.unique(Q):
            n = int((Q == q).sum())
            if q > Q_max:                                     # truncation branch
                idx = rng.choice(len(p), size=n, p=p)
                g = obj.grad(R[idx])
                cost += n
            else:
                Bk = 2 ** int(q)
                half = Bk // 2
                first = rng.choice(len(p), size=n, p=p)       # explicit 1st traj
                c_h1 = (rng.multinomial(half - 1, p, size=n)
                        if half > 1 else np.zeros((n, len(p)), dtype=np.int64))
                c_h1[np.arange(n), first] += 1                # 1st half incl. traj 1
                c_h2 = rng.multinomial(half, p, size=n)       # 2nd half
                J1 = R[first]
                Jhalf = (c_h1 @ R) / half
                Jfull = ((c_h1 + c_h2) @ R) / Bk
                g = obj.grad(J1) + Bk * (obj.grad(Jfull) - obj.grad(Jhalf))
                cost += n * Bk
            s1 += g.sum(axis=0)
            s2 += (g * g).sum(axis=0)
        done += r
    mean = s1 / reps
    var = np.maximum(s2 / reps - mean ** 2, 0.0)
    dev = np.abs(mean - obj.grad(J_true))
    m = int(np.argmax(dev))
    return float(dev[m]), float(np.sqrt(var[m] / reps)), cost / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3_000_000,
                    help="estimator replications per configuration (default 3e6)")
    args = ap.parse_args()

    env = FruitTreeEnv(depth=DEPTH, seed=42)
    # reward emitted at t = depth-1, hence the gamma^(depth-1) discount
    R = env.leaf_values / REWARD_SCALE * (GAMMA ** (DEPTH - 1))   # (2^d, M) in [0,1]
    p = frozen_policy_leaf_probs(DEPTH, POLICY_SEED)
    J_true = p @ R
    sigma = np.sqrt(p @ (R - J_true) ** 2)
    obj = LipschitzOnlyUtility(c=J_true.copy())                          # kink ON J^pi

    print(f"exact J^pi = {np.round(J_true, 4)}")
    print(f"per-channel return std sigma = {np.round(sigma, 4)}")
    print(f"reps per configuration = {args.reps:,}\n")

    rng = np.random.default_rng(DATA_SEED)
    rows = []
    t0 = time.time()
    for B in B_GRID:
        vb, vse = plugin_stats(obj, p, R, J_true, B, args.reps, rng)
        mb, mse, mc = mlmc_stats(obj, p, R, J_true, B, args.reps, rng)
        rows.append((B, vb, B, mb, mc, vse, mse))
        print(f"B={B:5d} | vanilla_npg bias {vb:.4e} (SE {vse:.1e}) at cost {B:5d}"
              f" | mlmc_npg bias {mb:.4e} (SE {mse:.1e}) at E[cost] {mc:5.2f}"
              f"  [{time.time()-t0:5.0f}s]", flush=True)

    A = np.array(rows, float)
    out = HERE.parent / "samples_to_eps_lipschitz.csv"
    np.savetxt(out, A, delimiter=",",
               header="B,vanilla_npg_bias,vanilla_npg_cost,mlmc_npg_bias,"
                      "mlmc_npg_cost,vanilla_npg_se,mlmc_npg_se",
               comments="")
    print(f"\nwrote {out}")

    slope = np.polyfit(np.log(A[:, 0]), np.log(A[:, 1]), 1)[0]
    print(f"vanilla_npg bias ~ B^{slope:.3f}  (theory: -1/2 at the kink; "
          f"cost to reach accuracy eps ~ eps^{1/-slope:.2f})")


if __name__ == "__main__":
    main()
