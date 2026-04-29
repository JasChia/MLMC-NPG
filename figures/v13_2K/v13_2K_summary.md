# v13 (2K) — doubled K with same B

## Hyperparameters

Same as v13 except all `K` doubled:

| algorithm | per-iter trajectories | outer K | total ≈ |
|---|---|---|---|
| MLMC-NPG | E[B_k] + N_inner·B_inner ≈ 144 | **70** | **10,084** |
| Bai-PG | 8 | **1,224** | **9,792** |
| REINFORCE / scalarized REINFORCE | 4 | **2,450** | **9,800** |

All other knobs identical to v13: `B_max=16, B_inner=4, N_inner=35, β=0.05, lr=0.05, γ=0.9999, δ=0.5`.

## DST tabular plateau (3 seeds)

| env | MLMC | Bai-PG | R-paperA | R-scalar |
|---|---|---|---|---|
| dst_open (10×5) | 8.77 ± 0.61 | 9.01 ± 0.00 | **9.76 ± 0.91** | 8.39 ± 0.11 |
| dst_floor (10×5) | 10.28 ± 1.20 | **10.79 ± 2.15** | 9.32 ± 0.90 | 8.54 ± 0.32 |
| dst_large_open (20×10) | 11.34 ± 0.08 | **13.74 ± 3.37** | 12.19 ± 0.77 | 11.45 ± 0.00 |
| dst_large_floor (20×10) | 11.92 ± 0.72 | **14.56 ± 3.97** | 13.76 ± 0.74 | 11.45 ± 0.00 |
| dst_original_open (10×11) | 21.70 ± 2.22 | 24.38 ± 0.12 | 23.08 ± 2.28 | **24.48 ± 0.03** |
| dst_original_floor (10×11) | 21.56 ± 2.58 | 24.27 ± 0.07 | 24.24 ± 0.07 | **24.49 ± 0.06** |

Key change vs 1K: MLMC moved up by 0.5–2.5 points across the board (more outer Adam steps helped). On dst_floor (small) MLMC is now within sd of Bai-PG. Scalarized REINFORCE retains its lead on the 10×11 envs.

## DST NN-32 plateau (3 seeds)

NN policy converges fast and plateaus early — results are largely unchanged from v13 (1K), confirming that NN-32 hit its capacity ceiling around 5k traj already.

| env | MLMC | Bai-PG | R-paperA | R-scalar |
|---|---|---|---|---|
| dst_open | 8.26 ± 0.00 | 8.26 ± 0.00 | 1.50 ± 0.00 | **8.33 ± 0.12** |
| dst_floor | 8.26 ± 0.00 | 8.26 ± 0.00 | **8.61 ± 0.36** | 8.33 ± 0.12 |
| dst_large_open | 11.14 ± 0.02 | 11.28 ± 0.15 | **12.78 ± 2.51** | 11.20 ± 0.00 |
| dst_large_floor | 8.75 ± 4.18 | 11.28 ± 0.15 | **12.46 ± 1.68** | 11.20 ± 0.00 |
| dst_original_open | 15.36 ± 0.07 | 18.45 ± 5.37 | 9.74 ± 12.92 | **20.66 ± 4.77** |
| dst_original_floor | 15.22 ± 0.18 | 18.45 ± 5.37 | 9.74 ± 12.92 | **21.53 ± 5.35** |

Paper-A REINFORCE collapsed deeper on dst_open at 2K (1.50 vs 4.81 at 1K) — more iterations giving NaN gradient more time to corrupt θ.

## Fruit Tree plateau by α (3 seeds, tabular)

| α | MLMC | Bai-PG | R-paperA | R-scalar |
|---|---|---|---|---|
| 0.0 | 47.51 ± 2.66 | 46.67 ± 3.37 | 43.59 ± 4.92 | **49.65 ± 2.61** |
| 0.25 | 37.46 ± 1.31 | 37.49 ± 1.53 | 35.31 ± 3.03 | **38.85 ± 1.56** |
| 0.5 | 33.71 ± 0.74 | 33.16 ± 1.54 | 32.24 ± 1.87 | **34.35 ± 1.02** |
| 1.0 | 12.31 ± 0.23 | 12.31 ± 0.21 | 11.82 ± 0.71 | **12.59 ± 0.37** |
| 2.0 | −0.80 ± 0.03 | **−0.78 ± 0.02** | −0.85 ± 0.11 | −0.72 ± 0.04 |

MLMC took the lead at α=0 (47.51 vs Bai 46.67), tied at α=0.25/0.5/1.0. Scalarized REINFORCE still wins at every α except 2.0.

## Verdict

Doubling K helps MLMC catch up on tabular DST (especially the smaller variants where K=35 was severely undertrained). But:
- MLMC still doesn't beat Bai-PG cleanly anywhere.
- Scalarized REINFORCE remains the leader on `dst_original_*` and Fruit Tree.
- NN-32 doesn't benefit from more iterations — capacity-bound.

The K=35 → K=70 doubling was the cheap fix. To get MLMC actual wins requires changing **what** the inner loop does (Fisher conditioning, β, NPG step size schedule, env structure), not just **how many** outer steps.

## Files

- `results/sweep_v13_dst_tab_2K/`, `results/sweep_v13_dst_nn32_2K/`, `results/sweep_v13_fruit_tree_alphas_2K/`
- `figures/v13_2K/dst_tab_fair_budget.png`, `figures/v13_2K/dst_nn32_fair_budget.png`
- `figures/v13_2K/fruit_tree_alphas_fair_budget.png`, `figures/v13_2K/fruit_tree_alphas_plateau_vs_alpha.png`
- `figures/v13_2K/trajectories/{6 envs} × {tab, nn32}_learned.png`
- `figures/v13_2K/v13_2K_pvalues.txt`
