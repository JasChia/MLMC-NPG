# v13 — alpha-fairness Fruit Tree + 6-variant DST + NN-32 DST

## Hyperparameters (fair budget ~5,000 trajectories)

| algorithm | per-iter trajectories | outer K | total ≈ |
|---|---|---|---|
| MLMC-NPG (`ganesh_mlmc`) | E[B_k] + N_inner·B_inner = 4.0625 + 35·4 ≈ **144.06** | **35** | **5,042** |
| Bai-PG (`bai_pg`)        | N1 + N2 = 4 + 4 = **8**                                  | **612** | **4,896** |
| REINFORCE (`reinforce`)  | N = **4**                                                | **1,225** | **4,900** |

MLMC: `B_max = 16`, `B_inner = 4`, `N_inner = 35`, `beta = 0.05`.
All runs use `gamma = 0.9999`, `lr = 0.05`, `delta = 0.5`, Adam ascent.

The expected MLMC outer batch with `B_max=16`:
`E[B_k] = 4·1 + 1/16 ≈ 4.0625` (one per level q=1..4, each contributes 2^q · 2⁻q = 1, plus the 1-traj fallback weighted by P(Q>4) = 1/16).

## What was run

| sweep | env(s) | policy | seeds | runs |
|---|---|---|---|---|
| `results/sweep_v13_dst_tab` | 6 DST variants | tabular | 0–2 | 54 |
| `results/sweep_v13_dst_nn32` | 6 DST variants | NN(32) | 0–2 | 54 |
| `results/sweep_v13_fruit_tree_alphas` | Fruit Tree depth=7 | tabular | 0–2 | 45 |

DST variants (3 maps × {open, floor}):
- `dst_open`           — 10×5, no sea floor
- `dst_floor`          — 10×5, Vamplew sea floor
- `dst_large_open`     — 20×10, no sea floor
- `dst_large_floor`    — 20×10, Vamplew sea floor
- `dst_original_open`  — 10×11 Vamplew layout, no sea floor
- `dst_original_floor` — 10×11 Vamplew layout, sea floor (canonical Vamplew 2011)

Fruit Tree alpha-fairness sweep at `α ∈ {0, 0.25, 0.5, 1, 2}`:
- α = 0: utilitarian sum (linear, not strictly concave — MLMC has no leverage)
- α = 0.25: between linear and √
- α = 0.5: 2·√x fairness  (= twice v11's `FruitTreeObj`)
- α = 1: log/proportional fairness
- α = 2: −1/x fairness (TCP-style)

Per-channel utility: `U_α(x) = x^(1−α)/(1−α)` (or `log(x)` at α=1).
Scalarization: `f(J) = Σ_m U_α(max(0, J_m) + δ)`. See `alt/objectives_alpha_fair.py`.

## Code

| file | role |
|---|---|
| `alt/objectives_alpha_fair.py` | parameterized α-fairness `AlphaFairObj` and DST-style `AlphaFairTimeObj` |
| `alt/run_v13_dst.py` | DST runner; supports tabular & NN policies, all 6 variants, REINFORCE reward-transform fix |
| `alt/run_v13_fruit_tree_alphas.py` | Fruit Tree α-sweep runner |
| `alt/plot_v13_dst.py` | 6-panel DST fair-budget plot, plateau CSV |
| `alt/plot_v13_fruit_tree_alphas.py` | per-alpha curves, plateau-vs-alpha line plot, CSV |
| `alt/compute_pvalues_v13.py` | rank-sum p-values for v13 |

## Outputs

- `figures/v13/dst_tab_fair_budget.png` — 6-panel DST tabular curves
- `figures/v13/dst_tab_plateaus.csv` — DST tabular plateau means/sd
- `figures/v13/dst_nn32_fair_budget.png` — 6-panel DST NN-32 curves
- `figures/v13/dst_nn32_plateaus.csv` — DST NN-32 plateau summary
- `figures/v13/fruit_tree_alphas_fair_budget.png` — per-alpha curves (5 panels)
- `figures/v13/fruit_tree_alphas_plateau_vs_alpha.png` — line plot
- `figures/v13/fruit_tree_alphas_plateaus.csv`
- `figures/v13/v13_pvalues.txt` — rank-sum table
