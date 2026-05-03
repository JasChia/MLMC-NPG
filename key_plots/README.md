# key_plots — parameters not shown in plots

## Shared

- Adam: `lr = 0.05`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`
- MLMC inner SGD step: `β = 0.05`
- Objective shift: `δ = 0.5`
- Policy: NN-32 single hidden layer, tanh activation, softmax over actions
- All algos run under **matched total trajectory budget**: `K_pg = T / (2B)`, `K_rein = T / B` where `T = K · per_iter_traj_mlmc(B, N)`
- Linear-scalarization REINFORCE uses uniform weights `w = (1/M, …, 1/M)`
- All sweeps GPU-accelerated (CUDA), 20 seeds each
- Per-seed JSONs include full f-curve, J-trajectory, final NN weights, and config — see `data/`

---

## Fruit Tree plots — `ft_nn32_B16_NK40_a*_fair_budget.{png,tikz}`

- env: depth-7 binary tree, M=6 nutrient channels, leaf-only reward
- env data: leaf-vector seed = 42, `np.random.uniform(0.3, 1.0, (128, 6)) * 10.0` (deterministic given action sequence)
- horizon: H = 7 (one step per tree level)
- discount: `γ = 0.99`
- objective: α-fairness `f(J) = Σ_m sign(α-1) · (max(0, J_m) + δ)^{1-α} / (1-α)` at `α ∈ {0.10, 0.25, 0.50, 1.00}` (log form when α=1)
- algos: MLMC-NPG, Policy Gradient (Bai-PG), REINFORCE-linear-scalarization
- `K_use`: MLMC=40, Bai=405, REINFORCE=810
- raw data: `data/fruit_tree/` (3 algos × 4 α × 20 seeds = 240 JSONs used; the folder also contains 80 unused `scalarized_reinforce_*` JSONs from earlier runs)

## BB-short plot — `bbs_T200_4algo_K40.{png,tikz}`

- env: 3 cells, 4 bottle states, 3 actions, M=3 reward channels (water, breakage, time)
- env params: `BREAK_PROB_PER_MOVE = 0.01`, `T_MAX = 200`, water +1 on delivery, breakage −1 on shatter, time −1 per active step
- horizon: H = T_MAX = 200
- discount: `γ = 0.9999`
- objective: `f(J) = √(max(0, J₀) + δ) + √(max(0, T_max + J₁) + δ) + √(max(0, T_max + J₂) + δ)`
- algos: MLMC-NPG, Policy Gradient (Bai-PG), REINFORCE-paper-A (Bai 2023 eq. 38, `delta_guard = 1.0`)
- `K_use`: MLMC=200, Bai=2039, REINFORCE=4078 (raw runs are at K=200 outer; plot truncates each curve at MLMC's iter-40 cumulative-trajectory budget ≈ 6,528 trajectories)
- raw data: `data/bb_short/` (3 algos × 20 seeds = 60 JSONs used; the folder also contains 20 unused `scalarized_reinforce_*` JSONs from earlier runs)

---

## Regenerating

```
cd key_plots/scripts
python plot_ft.py
python plot_bb_short.py
```

Requires `numpy`, `scipy`, `matplotlib`, `tikzplotlib`.
