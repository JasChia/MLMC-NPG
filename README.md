# MLMC-NPG — Figures Reproduction Kit

Reproduction code and data for the key figures in:

> **Breaking the Bias Barrier in Concave Multi-Objective Reinforcement Learning.**
> [arXiv:2603.08518](https://doi.org/10.48550/arXiv.2603.08518)

The Policy Gradient and REINFORCE baselines are taken from:

> Bai, Bedi, Aggarwal et al. **Joint Optimization of Concave Scalarized
> Multi-Objective Reinforcement Learning with Policy Gradient Based Algorithm.**
> *Journal of Artificial Intelligence Research.*
> [doi:10.1613/jair.1.13981](https://doi.org/10.1613/jair.1.13981)

---

## Repository Layout

```
MLMC-NPG/
├── src/                              # Algorithm + environment library
│   ├── algorithms.py                 # bai_pg_step, ganesh_mlmc_step, reinforce_step
│   ├── train.py                      # Outer Adam ascent loop
│   ├── envs/
│   │   ├── fruit_tree.py             # Depth-7 binary tree (M=6)
│   │   └── breakable_bottles_short.py # Breakable Bottles, T=200 (M=3)
│   ├── objectives/
│   │   └── alpha_fair.py             # α-fairness scalarization
│   └── policies/
│       └── nn_state.py               # NN-32 (1 hidden, tanh, softmax)
│
└── key_plots/                        # Paper-ready figures and reproduction scripts
    ├── bbs_b_sweep_4algo.tikz        # Breakable Bottles Short, B sweep
    ├── bbs_T200_4algo_K40.tikz       # Breakable Bottles Short, T=200 budget
    ├── ft_alpha_sweep_3algo.tikz     # Fruit Tree, α ∈ {0.10, 0.25, 0.50, 1.00}
    ├── ft_nn32_B16_NK40_a*.tikz      # Per-α Fruit Tree plots
    ├── *.png                         # Companion PNG previews
    ├── data/
    │   ├── bb_short/                 # Breakable Bottles Short JSONs (3 algos × 20 seeds)
    │   └── fruit_tree/               # Fruit Tree JSONs (3 algos × 4 α × 20 seeds)
    ├── scripts/
    │   ├── plot_bb_short.py          # → bbs_T200_4algo_K40.{png,tikz}
    │   ├── plot_ft.py                # → ft_nn32_B16_NK40_a*.{png,tikz}
    │   └── generate_ft_tikz.py       # → ft_alpha_sweep_3algo.tikz
    └── README.md                     # Per-plot hyperparameter notes
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy torch scipy matplotlib tikzplotlib
```

A GPU (CUDA) is not required to regenerate the figures from existing JSON data;
it matters only for rerunning the experiments themselves.

### 2. Regenerate the Breakable Bottles Short figure

```bash
cd key_plots/scripts
python plot_bb_short.py        # → ../bbs_T200_4algo_K40.{png,tikz}
```

The `bbs_b_sweep_4algo.tikz` figure is committed as a finalized artifact.
To regenerate it from scratch you would need to run the experiment runner
(see [Running Experiments](#running-experiments)) for `B ∈ {4, 8, 16}` and
adapt `plot_bb_short.py` as a template for the multi-panel grouped plot.

### 3. Regenerate the Fruit Tree figures

```bash
cd key_plots/scripts
python plot_ft.py              # → ../ft_nn32_B16_NK40_a{010,025,050,100}_fair_budget.{png,tikz}
python generate_ft_tikz.py     # → ../ft_alpha_sweep_3algo.tikz
```

`generate_ft_tikz.py` produces the 2×2 grouped plot directly without `tikzplotlib`.

---

## Figures Guide

### `bbs_b_sweep_4algo.tikz` — Breakable Bottles Short, B sweep

A 1×2 grouped plot plus a third standalone panel showing how the algorithms scale
with the inner batch size B.

| Panel | B  | K_mlmc |
|-------|----|--------|
| (a)   | 4  | 40     |
| (b)   | 8  | 40     |
| (c)   | 16 | 40     |

- **Environment**: Breakable Bottles Short, T=200
- **Objective**: `f(J^π) = √(max(0, J^π₀)+δ) + √(max(0, T+J^π₁)+δ) + √(max(0, T+J^π₂)+δ)`
  - `J^π₀` = water reward (delivery), `J^π₁` = breakage penalty, `J^π₂` = time penalty
- **Algorithms**: MLMC-NPG (purple), Policy Gradient (blue), REINFORCE (red)
- **Seeds**: 20 per (algorithm, B)

### `bbs_T200_4algo_K40.tikz` — Breakable Bottles Short, T=200 budget

- Same environment and objective as above, fixed `B = 8`
- Curves are truncated at MLMC-NPG's K=40 cumulative-trajectory budget (~6,528 trajectories)
- **K used**: MLMC-NPG=200, Policy Gradient=2039, REINFORCE=4078 (all truncated to the same budget)
- **Data**: `key_plots/data/bb_short/`
- **Script**: `key_plots/scripts/plot_bb_short.py`

### `ft_alpha_sweep_3algo.tikz` — Fruit Tree, α ∈ {0.10, 0.25, 0.50, 1.00}

A 2×2 grouped plot showing MLMC-NPG vs. Policy Gradient vs. REINFORCE across four
α-fairness regimes.

| Panel | α    | Regime                |
|-------|------|-----------------------|
| (a)   | 0.10 | Near-linear fairness  |
| (b)   | 0.25 | Mild concavity        |
| (c)   | 0.50 | Canonical √ fairness  |
| (d)   | 1.00 | Log fairness          |

- **Environment**: Fruit Tree, depth 7, M=6 nutrient channels
- **Objective**: α-fairness `f_α(J^π) = Σ_m (J^π_m)^{1−α} / (1−α)` (log form when α=1)
- **K used**: MLMC-NPG=40, Policy Gradient=405, REINFORCE=810 (budget-matched)
- **Data**: `key_plots/data/fruit_tree/`
- **Script**: `key_plots/scripts/generate_ft_tikz.py`

---

## Running Experiments

The `src/` library implements the three algorithms on a shared rollout / policy
/ objective interface. A minimal driver looks like:

```python
import numpy as np, torch
from src.envs.fruit_tree import FruitTreeEnv, fruit_tree_batch_rollout
from src.objectives.alpha_fair import AlphaFairObj
from src.policies.nn_state import NNStatePolicy
from src.train import train

rng = np.random.default_rng(0)
env = FruitTreeEnv(depth=7)
obj = AlphaFairObj(alpha=0.5, delta=0.5)
policy = NNStatePolicy(num_states=env.num_states, num_actions=2, hidden=32)

def rollout_fn(H, B):
    return fruit_tree_batch_rollout(env.config, policy, H, B, rng)

history = train(
    algorithm="ganesh_mlmc",                 # or "bai_pg" / "reinforce"
    policy=policy, rollout_fn=rollout_fn,
    obj=obj, gamma=0.99, K=40, H=7,
    rng=rng, lr=0.05,
    algo_kwargs={"B_max": 16, "B_inner": 16, "N_inner": 20, "beta": 0.05},
)
```

The same pattern applies to Breakable Bottles Short by swapping the env, rollout
function, and objective (`BreakableBottlesShortObj` defined in
`src/envs/breakable_bottles_short.py`).

---

## Data Format

Each run produces one JSON file per `(algorithm, environment, config, seed)` with
the schema:

```json
{
  "algo":         "ganesh_mlmc",
  "env":          "fruit_tree",
  "config":       { "B": 16, "N": 20, "K": 40, "alpha": 0.10, ... },
  "f":            [<float>, ...],          // scalarized objective at each iteration
  "J_hat":        [[<float>, ...], ...],   // per-objective return estimates
  "trajectories": [<int>, ...],            // cumulative trajectory count at each iter
  "seed":         0
}
```

Filename pattern:
```
<algo>_<env>_<params>_seed<N>.json
```

---

## Algorithm Reference

| Code name      | Reference        | Description                                              |
|----------------|------------------|----------------------------------------------------------|
| `ganesh_mlmc`  | this work, Alg. 2 | **MLMC-NPG**: multi-level MC gradient + NPG ascent       |
| `bai_pg`       | Bai et al., Alg. 1 | **Policy Gradient**: biased plug-in estimator             |
| `reinforce`    | Bai et al., eq. 38 | **REINFORCE**: per-step training-reward baseline          |

### Shared hyperparameters

| Parameter | Value | Description                                          |
|-----------|-------|------------------------------------------------------|
| Optimizer | Adam  | lr=0.05, β₁=0.9, β₂=0.999, ε=1e-8                    |
| MLMC β    | 0.05  | Inner SGD step size for MLMC-NPG                     |
| δ         | 0.5   | Additive guard inside the α-fairness scalarization   |
| Policy    | NN-32 | One hidden layer, 32 units, tanh, softmax over actions |
| Seeds     | 20    | Independent seeds per (algorithm, configuration)     |

### Budget matching

To compare algorithms fairly, all three are run for the same total trajectory
budget `T`:

```
T      = K_mlmc × per_iter_traj(B, N)   # MLMC-NPG total budget
K_pg   = T / (2B)                       # Policy Gradient uses 2B trajectories per iter
K_rein = T / B                          # REINFORCE uses B trajectories per iter
```

---

## Environment Reference

### Fruit Tree (depth-7 binary tree)

- **States**: tree nodes, indexed 0…(2^d − 1)
- **Actions**: 2 (left = 0, right = 1)
- **Reward**: M = 6 nutrient channels emitted only at the leaf (last step)
- **Leaf rewards**: `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0`, seed = 42
- **Horizon**: H = 7, discount γ = 0.99
- **Objective**: α-fairness on the leaf return vector

### Breakable Bottles Short (T = 200)

- **States**: 3 cells × 4 bottle states + a terminal sink
- **Actions**: 3 (left, right, deliver/load)
- **Reward**: M = 3 — water delivery (+1), bottle breakage (−1), time (−1 / step)
- **Dynamics**: `BREAK_PROB_PER_MOVE = 0.01` while carrying bottles
- **Horizon**: H = T_MAX = 200, discount γ = 0.9999
- **Objective**: `f(J^π) = √(J^π₀+δ) + √(T+J^π₁+δ) + √(T+J^π₂+δ)`

---

## Notation

| Symbol      | Meaning                                          |
|-------------|--------------------------------------------------|
| `J^π`       | Expected return vector under policy π            |
| `J^π_m`     | Component m of `J^π` (objective m)               |
| `f_α(J^π)`  | α-fairness scalarization of `J^π`                |
| `B`         | Batch size (trajectories per inner loop)         |
| `N`         | Number of inner-loop iterations (MLMC levels)    |
| `K`         | Number of outer (policy update) iterations       |
| `M`         | Number of reward / objective channels            |
| `H`         | Episode horizon                                  |
