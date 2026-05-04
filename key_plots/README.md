# key_plots — hyperparameters not shown in the figures

## Shared

- Adam: `lr = 0.05`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`
- MLMC inner SGD step: `β = 0.05`
- Objective shift: `δ = 0.5`
- Policy: NN-32 single hidden layer, tanh activation, softmax over actions
- Matched total trajectory budget: `K_pg = T / (2B)`, `K_rein = T / B`,
  with `T = K_mlmc · per_iter_traj_mlmc(B, N)`
- All sweeps run on GPU (CUDA), 20 seeds each
- Each per-seed JSON contains the full f-curve, J-trajectory, final NN weights, and config

---

## Fruit Tree plots — `ft_nn32_B16_NK40_a*_fair_budget.{png,tikz}` and `ft_alpha_sweep_3algo.tikz`

- env: depth-7 binary tree, M=6 nutrient channels, leaf-only reward
- leaf vectors: seeded with `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0`, seed=42
  (deterministic given the action sequence)
- horizon: H = 7 (one step per tree level)
- discount: `γ = 0.99`
- objective: α-fairness `f(J^π) = Σ_m (max(0, J^π_m) + δ)^{1−α} / (1−α)`
  at `α ∈ {0.10, 0.25, 0.50, 1.00}` (log form when α=1)
- algos: MLMC-NPG, Policy Gradient, REINFORCE
- `K` per algorithm: MLMC-NPG=40, Policy Gradient=405, REINFORCE=810
- raw data: `data/fruit_tree/` (3 algos × 4 α × 20 seeds = 240 JSONs)

## Breaking Bottles Short plot — `bbs_T200_4algo_K40.{png,tikz}`

- env: 3 cells, 4 bottle states, 3 actions, M=3 reward channels (water, breakage, time)
- env params: `BREAK_PROB_PER_MOVE = 0.01`, `T_MAX = 200`,
  water +1 on delivery, breakage −1 on shatter, time −1 per active step
- horizon: H = T_MAX = 200
- discount: `γ = 0.9999`
- objective: `f(J^π) = √(max(0, J^π₀) + δ) + √(max(0, T_max + J^π₁) + δ) + √(max(0, T_max + J^π₂) + δ)`
- algos: MLMC-NPG, Policy Gradient, REINFORCE (`delta_guard = 1.0` for REINFORCE)
- `K` per algorithm: MLMC-NPG=200, Policy Gradient=2039, REINFORCE=4078
  (raw runs use K=200 outer; the plot truncates each curve at MLMC-NPG's iter-40
  cumulative-trajectory budget ≈ 6,528 trajectories)
- raw data: `data/bb_short/` (3 algos × 20 seeds = 60 JSONs)

---

## Regenerating

```
cd key_plots/scripts
python plot_ft.py            # → 4 individual α plots
python generate_ft_tikz.py   # → ft_alpha_sweep_3algo.tikz
python plot_bb_short.py      # → bbs_T200_4algo_K40 plot
```

Requires `numpy`, `scipy`, `matplotlib`, `tikzplotlib`.
