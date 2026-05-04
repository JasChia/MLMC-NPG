# MLMC-NPG — Figures Reproduction Kit

Reproduction code and data for the key figures in **"Multi-Level Monte Carlo Natural Policy Gradient for Multi-Objective Reinforcement Learning"** (MLMC-NPG paper, referred to throughout as *Paper B*).

---

## Repository Layout

```
MLMC-NPG/
├── src/                         # Algorithm + environment library
│   ├── algorithms.py            # Core algorithm steps (all three algos)
│   ├── train.py                 # Outer Adam training loop
│   ├── envs/                    # Environments
│   │   ├── fruit_tree.py        # Depth-7 binary fruit tree (M=6)
│   │   ├── breakable_bottles_short.py  # Breaking Bottles Short (T=200, M=3)
│   │   ├── dst/                 # Deep Sea Treasure variants
│   │   ├── cliff_walking.py
│   │   ├── mo_mountain_car.py
│   │   └── ...
│   ├── objectives/
│   │   └── alpha_fair.py        # α-fairness objective + gradient
│   └── policies/
│       ├── nn_state.py          # NN-32 (1 hidden, tanh, softmax) policy
│       ├── tabular.py           # Tabular softmax policy
│       └── log_linear.py        # Log-linear policy
│
├── key_plots/                   # Paper-ready figures (the main output)
│   ├── bbs_b_sweep_4algo.tikz   # Fig: BB-Short B∈{4,8,16} sweep, 4 algos  ← PAPER FIGURE
│   ├── bbs_T200_4algo_K40.tikz  # Fig: BB-Short T=200 K=40 budget, 3 algos  ← PAPER FIGURE
│   ├── ft_alpha_sweep_3algo.tikz # Fig: Fruit Tree α∈{0.10,0.25,0.50,1.00}, 3 algos  ← PAPER FIGURE
│   ├── ft_nn32_B16_NK40_a*.tikz # Individual per-α Fruit Tree plots
│   ├── *.png                    # Companion PNG previews
│   ├── data/
│   │   ├── bb_short/            # 80 JSONs — BB-Short (20 seeds × 4 algos)
│   │   └── fruit_tree/          # 320 JSONs — Fruit Tree (20 seeds × 4 α × 4 algos)
│   ├── scripts/
│   │   ├── plot_bb_short.py     # Regenerates bbs_T200_4algo_K40.{png,tikz}
│   │   ├── plot_ft.py           # Regenerates ft_nn32_B16_NK40_a*.{png,tikz}
│   │   ├── plot_ft_combined.py  # Regenerates ft_alpha_sweep_3algo (via tikzplotlib)
│   │   ├── generate_ft_tikz.py  # Regenerates ft_alpha_sweep_3algo (pure Python, preferred)
│   │   └── strip_algo_from_tikz.py  # Utility: remove an algorithm's curves from a tikz
│   └── README.md                # Per-plot hyperparameter notes
│
└── legacy/                      # Archived exploratory code — NOT tracked by git
    ├── alt/                     # Earlier experiment runners and plot scripts (v10–v17)
    └── ...
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy torch scipy matplotlib tikzplotlib
```

A GPU (CUDA) is not required for the plotting scripts; it matters only for rerunning experiments.

### 2. Regenerate the Breaking Bottles Short figures

```bash
cd key_plots/scripts
python plot_bb_short.py        # → ../bbs_T200_4algo_K40.{png,tikz}
```

> **Note on `bbs_b_sweep_4algo.tikz`**: this figure (B ∈ {4, 8, 16} sweep) was produced by
> an external matplotlib script and hand-edited. The tikz file itself is the canonical
> artifact and is committed as-is. To regenerate it from scratch you would need to run
> the B-sweep experiment (see [Running Experiments](#running-experiments)) and then use
> `plot_bb_short.py` as a template.

### 3. Regenerate the Fruit Tree figures

```bash
cd key_plots/scripts
python plot_ft.py              # → ../ft_nn32_B16_NK40_a{010,025,050,100}_fair_budget.{png,tikz}
python generate_ft_tikz.py    # → ../ft_alpha_sweep_3algo.tikz  (no tikzplotlib needed)
# or:
python plot_ft_combined.py    # → ../ft_alpha_sweep_3algo.{png,tikz}  (requires tikzplotlib)
```

---

## Figures Guide

### `bbs_b_sweep_4algo.tikz` — Breaking Bottles Short, B sweep

| Panel | B | K_mlmc | K_pg | K_rein |
|-------|---|--------|------|--------|
| (a)   | 4 | 40     | auto | auto   |
| (b)   | 8 | 40     | auto | auto   |
| (c)   | 16| 40     | auto | auto   |

- **Environment**: Breaking Bottles Short (`breakable_bottles_short.py`), T=200
- **Objective**: `f(J^π) = √(max(0,J^π₀)+δ) + √(max(0,T+J^π₁)+δ) + √(max(0,T+J^π₂)+δ)`
  - J^π₀ = water reward (delivery), J^π₁ = breakage penalty, J^π₂ = time penalty
- **Algorithms**: MLMC-NPG (purple), Policy Gradient (blue), REINFORCE (red), scalarized REINFORCE (orange)
- **Seeds**: 20 per (algo, B)
- **Data**: `key_plots/data/bb_short/` — B=8 files (`*B8*`); B=4 and B=16 data from external sweep

### `bbs_T200_4algo_K40.tikz` — Breaking Bottles Short, T=200 trajectory budget

- Same environment and objective as above, fixed B=8
- **Budget cap**: curves truncated at MLMC-NPG's K=40 cumulative-trajectory count (~6,528 trajectories)
- **K used**: MLMC=200, Bai-PG=2039, REINFORCE=4078 (all truncated to same budget)
- **Data**: `key_plots/data/bb_short/` (80 JSONs: 4 algos × 20 seeds)
- **Script**: `key_plots/scripts/plot_bb_short.py`

### `ft_alpha_sweep_3algo.tikz` — Fruit Tree, α ∈ {0.10, 0.25, 0.50, 1.00}

2×2 grouped plot showing MLMC-NPG vs Policy Gradient vs REINFORCE across four α-fairness regimes.

| Panel | α    | Regime              |
|-------|------|---------------------|
| (a)   | 0.10 | Near-linear fairness |
| (b)   | 0.25 | Mild concavity       |
| (c)   | 0.50 | Canonical √ fairness |
| (d)   | 1.00 | Log fairness         |

- **Environment**: Fruit Tree, depth 7, M=6 nutrient channels
- **Objective**: α-fairness `f_α(J^π) = Σ_m (J^π_m)^{1−α} / (1−α)` (log form when α=1)
- **K used**: MLMC=40, Bai-PG=405, REINFORCE=810 (budget-matched)
- **Data**: `key_plots/data/fruit_tree/` (240 JSONs: 3 algos × 4 α × 20 seeds)
- **Script**: `key_plots/scripts/generate_ft_tikz.py` (preferred) or `plot_ft_combined.py`

---

## Running Experiments

The `src/` library provides the algorithm implementations; `legacy/alt/run_v15_search.py`
is the canonical experiment runner used to produce the data in `key_plots/data/`.

### Generating Fruit Tree data

```bash
# From the repo root:
python -m legacy.alt.run_v15_search \
  --out_dir key_plots/data/fruit_tree \
  --env_kind fruit_tree --alpha 0.10 \
  --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --beta 0.05 --lr 0.05 \
  --delta 0.5 --gamma 0.99 \
  --seeds 20 --workers 4
```

Repeat with `--alpha 0.25 / 0.50 / 1.00`. The runner auto-derives `K_pg` and `K_rein`
to match MLMC-NPG's total trajectory budget.

### Generating Breaking Bottles Short data

```bash
python -m legacy.alt.run_v15_search \
  --out_dir key_plots/data/bb_short \
  --env_kind breakable_bottles_short \
  --policy nn --hidden 32 \
  --B 8 --N 20 --K 200 --beta 0.05 --lr 0.05 \
  --seeds 20 --workers 4
```

---

## Data Format

Each run produces one JSON file per `(algo, env, config, seed)` with the schema:

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
e.g.  ganesh_mlmc_fruit_tree_a010_d7_nn32_B16_N20_K40_beta0.05_lr0.05_v15_ft_nn32_B16_NK40_a010_seed0.json
```

---

## Algorithm Reference

| Code name        | Paper ref          | Description                                              |
|------------------|--------------------|----------------------------------------------------------|
| `ganesh_mlmc`    | Paper B, Alg. 2    | **MLMC-NPG**: multi-level MC gradient + NPG ascent       |
| `bai_pg`         | Paper A, Alg. 1    | **Policy Gradient**: biased plug-in estimator (Bai et al.) |
| `reinforce`      | Paper A, eq. 38    | **REINFORCE**: per-step baseline, Paper A variant        |
| `scalarized_reinforce` | baseline    | REINFORCE on linear scalarization (uniform weights)      |

*Paper A* = Bai et al., multi-objective policy gradient baseline.  
*Paper B* = this work (MLMC-NPG).

### Shared hyperparameters (all experiments)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam  | lr=0.05, β₁=0.9, β₂=0.999, ε=1e-8 |
| MLMC β    | 0.05  | Inner SGD step size for Ganesh MLMC |
| δ (delta) | 0.5   | Objective shift in α-fairness guard |
| Policy    | NN-32 | 1 hidden layer, 32 units, tanh, softmax |
| Seeds     | 20    | All experiments use 20 independent seeds |
| GPU       | CUDA  | All runs accelerated on GPU |

### Budget matching

To ensure a fair comparison, all algorithms are given the same total trajectory budget `T`:

```
T = K_mlmc × per_iter_traj(B, N)      # MLMC-NPG budget
K_pg   = T / (2B)                      # Bai-PG uses 2B traj/iter
K_rein = T / B                         # REINFORCE uses B traj/iter
```

---

## Environment Reference

### Fruit Tree (depth-7 binary tree)

- **States**: tree nodes at each level (depth 0–7)
- **Actions**: left/right at each node (binary branching)
- **Reward**: M=6 nutrient channels, received only at leaves
- **Leaf rewards**: seeded with `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0`, seed=42
- **Horizon**: H=7, discount γ=0.99
- **Objective**: α-fairness over `J^π_m = γ^H · leaf_reward_m`

### Breaking Bottles Short (BB-Short, T=200)

- **States**: 3 cells × 4 bottle states × agent position = compact tabular
- **Actions**: 3 (left, right, deliver)
- **Reward**: M=3 channels — water delivery (+1), bottle breakage (−1), time (−1/step active)
- **Dynamics**: `BREAK_PROB_PER_MOVE = 0.01` per move
- **Horizon**: H=T_MAX=200, discount γ=0.9999
- **Objective**: `f(J^π) = √(J^π₀+δ) + √(T+J^π₁+δ) + √(T+J^π₂+δ)`

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `J^π`  | Expected return vector under policy π |
| `J^π_m` | m-th component of return (objective m) |
| `f_α(J^π)` | α-fairness scalarization of `J^π` |
| B | Batch size (trajectories per inner loop) |
| N | Number of inner loop iterations (MLMC levels) |
| K | Number of outer (policy update) iterations |
| M | Number of reward/objective channels |
| H | Episode horizon |

---

## Using the `strip_algo_from_tikz.py` Utility

To remove one algorithm's curves from any existing tikz figure (e.g., drop scalarized REINFORCE):

```bash
cd key_plots/scripts
python strip_algo_from_tikz.py ../bbs_T200_4algo_K40.tikz darkorange25512714 \
    --label "REINFORCE (scalar)" \
    --legend-columns 3 \
    --out ../bbs_T200_3algo_K40.tikz
```

Color names follow the `definecolor` declarations at the top of each tikz file.
