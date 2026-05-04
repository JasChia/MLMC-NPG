# key_plots — hyperparameters not shown in the figures

## Shared

- Adam: `lr = 0.05`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`
- MLMC inner SGD step: `β = 0.05`
- Objective shift: `δ = 0.5`
- Policy: NN-32 single hidden layer, tanh activation, softmax over actions
- Matched total trajectory budget across algorithms:
  `K_pg = T / (2B)`, `K_rein = T / B`, with `T = K_mlmc · per_iter_traj_mlmc(B, N)`
- 20 seeds per configuration; runs are GPU-accelerated
- Each per-seed JSON contains the full f-curve, J-trajectory, and config

---

## Fruit Tree — `fruit_tree_alpha_sweep.{png,tikz}`

- env: depth-7 binary tree, M = 6 nutrient channels, leaf-only reward
- leaf vectors: seeded with `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0`, seed = 42
- horizon: H = 7 (one step per tree level)
- discount: `γ = 0.99`
- objective: α-fairness `f(J^π) = Σ_m (max(0, J^π_m) + δ)^{1−α} / (1−α)`
  at `α ∈ {0.10, 0.25, 0.50, 1.00}` (log form when α = 1)
- algos: MLMC-NPG, Policy Gradient, REINFORCE
- per-algorithm K: MLMC-NPG = 40, Policy Gradient = 405, REINFORCE = 810
- raw data: `data/fruit_tree/` (3 algos × 4 α × 20 seeds = 240 JSONs)

## Breakable Bottles — `breakable_bottles_b_sweep.{png,tikz}`

- env: 3 cells, 4 bottle states, 3 actions, M = 3 reward channels (water, breakage, time)
- env params: `BREAK_PROB_PER_MOVE = 0.01`, `T_MAX = 200`,
  water +1 on delivery, breakage −1 on shatter, time −1 per active step
- horizon: H = T_MAX = 200
- discount: `γ = 0.9999`
- objective: `f(J^π) = √(max(0, J^π₀) + δ) + √(max(0, T_max + J^π₁) + δ) + √(max(0, T_max + J^π₂) + δ)`
- algos: MLMC-NPG, Policy Gradient, REINFORCE (`delta_guard = 1.0` for REINFORCE)
- B sweep across `{4, 8, 16}` (panels a, b, c respectively)
- raw data: `data/breakable_bottles/`

---

## Regenerating

Generate the raw experimental data (skips files that already exist):

```bash
cd key_plots/scripts
python run_experiments.py
```

Then regenerate the figures:

```bash
python plot_fruit_tree.py            # → ../fruit_tree_alpha_sweep.{tikz,png}
python plot_breakable_bottles.py     # → ../breakable_bottles_b_sweep.{tikz,png}
```

Requires `numpy`, `torch`, `matplotlib`. The plot scripts write `.tikz` directly
without `tikzplotlib`.
