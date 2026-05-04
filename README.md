# MLMC-NPG

Reproduction code and data accompanying an anonymous submission on
multi-level Monte Carlo natural policy gradient method for concave
multi-objective reinforcement learning. *Author and citation details are
withheld for double-blind review.*

The Policy Gradient and REINFORCE baselines follow the prior work cited
in the submission (a published JAIR article on concave-scalarized
multi-objective policy gradient).

---

## Repository Layout

```
MLMC-NPG/
├── src/                              # Algorithm + environment library
│   ├── algorithms.py                 # bai_pg_step, mlmc_npg_step, reinforce_step
│   ├── train.py                      # Outer Adam ascent loop
│   ├── envs/
│   │   ├── fruit_tree.py             # Depth-7 binary tree (M=6)
│   │   └── breakable_bottles.py      # Breakable Bottles, T=200 (M=3)
│   ├── objectives/
│   │   └── alpha_fair.py             # α-fairness scalarization
│   └── policies/
│       └── nn_state.py               # NN-32 (1 hidden, tanh, softmax)
│
└── key_plots/                        # Paper-ready figures and reproduction scripts
    ├── breakable_bottles_b_sweep.{tikz,png}   # Breakable Bottles, B ∈ {4, 8, 16}
    ├── fruit_tree_alpha_sweep.{tikz,png}      # Fruit Tree, α ∈ {0.10, 0.25, 0.50, 1.00}
    ├── data/
    │   ├── breakable_bottles/        # Per-seed JSONs (3 algos × 3 B values × 20 seeds)
    │   └── fruit_tree/               # Per-seed JSONs (3 algos × 4 α × 20 seeds)
    ├── scripts/
    │   ├── run_experiments.py        # Generates all data (calls src/ to train)
    │   ├── plot_breakable_bottles.py # → breakable_bottles_b_sweep.{tikz,png}
    │   └── plot_fruit_tree.py        # → fruit_tree_alpha_sweep.{tikz,png}
    └── README.md                     # Per-plot hyperparameter notes
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy torch scipy matplotlib
```

A GPU (CUDA) is not required to regenerate the figures from the existing JSON
data; it matters only for `run_experiments.py` (training is GPU-accelerated).

### 2. Regenerate the figures from existing data

```bash
cd key_plots/scripts
python plot_fruit_tree.py            # → ../fruit_tree_alpha_sweep.{tikz,png}
python plot_breakable_bottles.py     # → ../breakable_bottles_b_sweep.{tikz,png}
```

Each script writes both a `.tikz` (canonical, paper-ready) and a `.png` preview.

### 3. Regenerate the data from scratch

To produce the JSON inputs that the plot scripts consume:

```bash
cd key_plots/scripts
python run_experiments.py            # all envs, all configs, 20 seeds each
```

`run_experiments.py` skips any per-seed JSON that already exists; delete the
files (or pass a different `--seeds` count) to force a rerun. Useful flags:

```bash
python run_experiments.py --env fruit_tree
python run_experiments.py --env breakable_bottles --B 4 8 16
python run_experiments.py --seeds 5             # smoke test
```

After fresh data lands in `key_plots/data/`, rerun the two plot scripts in
step 2 to refresh the figures.

---

## Figures Guide

### `breakable_bottles_b_sweep.{tikz,png}` — Breakable Bottles, B sweep

A 1×2 grouped plot plus a third standalone panel showing how the algorithms scale
with the inner batch size B.

| Panel | B  | K_mlmc |
|-------|----|--------|
| (a)   | 4  | 40     |
| (b)   | 8  | 40     |
| (c)   | 16 | 40     |

- **Environment**: Breakable Bottles, T = 200
- **Objective**: `f(J^π) = √(max(0, J^π₀)+δ) + √(max(0, T+J^π₁)+δ) + √(max(0, T+J^π₂)+δ)`
  (`J^π₀` = water reward, `J^π₁` = breakage, `J^π₂` = time)
- **Algorithms**: MLMC-NPG (purple), Policy Gradient (blue), REINFORCE (red)
- **Seeds**: 20 per (algorithm, B)
- **Data**: `key_plots/data/breakable_bottles/`
- **Script**: `key_plots/scripts/plot_breakable_bottles.py`

### `fruit_tree_alpha_sweep.{tikz,png}` — Fruit Tree, α sweep

A 2×2 grouped plot showing MLMC-NPG vs. Policy Gradient vs. REINFORCE across four
α-fairness regimes.

| Panel | α    | Regime                |
|-------|------|-----------------------|
| (a)   | 0.10 | Near-linear fairness  |
| (b)   | 0.25 | Mild concavity        |
| (c)   | 0.50 | Canonical √ fairness  |
| (d)   | 1.00 | Log fairness          |

- **Environment**: Fruit Tree, depth 7, M = 6 nutrient channels
- **Objective**: α-fairness `f_α(J^π) = Σ_m (J^π_m)^{1−α} / (1−α)` (log form when α = 1)
- **K used**: MLMC-NPG = 40, Policy Gradient = 405, REINFORCE = 810 (budget-matched)
- **Data**: `key_plots/data/fruit_tree/`
- **Script**: `key_plots/scripts/plot_fruit_tree.py`

---

## Data Format

Each run produces one JSON file per `(algorithm, environment, config, seed)` with
the schema:

```json
{
  "algo":         "mlmc_npg",
  "env":          "fruit_tree",
  "config":       { "B": 16, "N": 20, "K": 40, "alpha": 0.10, ... },
  "f":            [<float>, ...],          // scalarized objective at each iteration
  "J_hat":        [[<float>, ...], ...],   // per-objective return estimates
  "trajectories": [<int>, ...],            // cumulative trajectory count at each iter
  "seed":         0
}
```

Filename pattern (produced by `run_experiments.py`):
```
<algo>_<env>_..._B<B>_..._seed<N>.json
```

---

## Algorithm Reference

| Code name      | Reference            | Description                                            |
|----------------|----------------------|--------------------------------------------------------|
| `mlmc_npg`     | this work, Alg. 2    | **MLMC-NPG**: multi-level MC gradient + NPG ascent     |
| `bai_pg`       | prior work, Alg. 1   | **Policy Gradient**: biased plug-in estimator          |
| `reinforce`    | prior work, eq. 38   | **REINFORCE**: per-step training-reward baseline       |

### Shared hyperparameters

| Parameter | Value | Description                                            |
|-----------|-------|--------------------------------------------------------|
| Optimizer | Adam  | lr = 0.05, β₁ = 0.9, β₂ = 0.999, ε = 1e-8              |
| MLMC β    | 0.05  | Inner SGD step size for MLMC-NPG                       |
| δ         | 0.5   | Additive guard inside the α-fairness scalarization     |
| Policy    | NN-32 | One hidden layer, 32 units, tanh, softmax over actions |
| Seeds     | 20    | Independent seeds per (algorithm, configuration)       |

### Budget matching

All three algorithms consume the same total trajectory budget `T`:

```
T      = K_mlmc × per_iter_traj(B, N)   # MLMC-NPG total budget
K_pg   = T / (2B)                       # Policy Gradient uses 2B trajectories per iter
K_rein = T / B                          # REINFORCE uses B trajectories per iter
```

---

## Environment Reference

### Fruit Tree (depth-7 binary tree)

- **States**: tree nodes, indexed 0 … (2^d − 1)
- **Actions**: 2 (left = 0, right = 1)
- **Reward**: M = 6 nutrient channels emitted only at the leaf (last step)
- **Leaf rewards**: `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0`, seed = 42
- **Horizon**: H = 7, discount γ = 0.99
- **Objective**: α-fairness on the leaf return vector

### Breakable Bottles (T = 200)

- **States**: 3 cells × 4 bottle states + a terminal sink
- **Actions**: 3 (left, right, deliver/load)
- **Reward**: M = 3 — water delivery (+1), bottle breakage (−1), time (−1 / step)
- **Dynamics**: `BREAK_PROB_PER_MOVE = 0.01` while carrying bottles
- **Horizon**: H = T_MAX = 200, discount γ = 0.9999
- **Objective**: `f(J^π) = √(J^π₀+δ) + √(T+J^π₁+δ) + √(T+J^π₂+δ)`

---

## Notation

| Symbol      | Meaning                                       |
|-------------|-----------------------------------------------|
| `J^π`       | Expected return vector under policy π         |
| `J^π_m`     | Component m of `J^π` (objective m)            |
| `f_α(J^π)`  | α-fairness scalarization of `J^π`             |
| `B`         | Batch size (trajectories per inner loop)      |
| `N`         | Number of inner-loop iterations (MLMC levels) |
| `K`         | Number of outer (policy update) iterations    |
| `M`         | Number of reward / objective channels         |
| `H`         | Episode horizon                               |
