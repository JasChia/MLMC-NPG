# Pre-final fruit-tree suite (NN-32, 20 seeds, GPU)

Regenerated 2026-05-02 using v15 fair-budget protocol on GPU.

## Configuration

```
env_kind   = fruit_tree (depth 7, M=6 nutrient channels)
policy     = NN-32 (one-hidden-layer MLP, tanh, softmax over 2 actions)
B          = 16, N = 20, K = 40   (matches existing v16 config)
gamma      = 0.99
lr         = 0.05
beta       = 0.05  (MLMC inner step)
delta      = 0.5   (objective shift)
seeds      = 20 each (compared with prior 3-7)
device     = cuda
algos      = MLMC-NPG | Bai-PG | REINFORCE (paper A) | Scalarized REINFORCE
objective  = α-fairness  f(J) = Σ_m sign(α-1) · (J_m+δ)^(1-α) / (1-α)
```

K_use translation (matched trajectory budget):
- MLMC: K=40 outer
- Bai: K_pg = 405 outer Adam steps
- REINFORCE (paper A): K_rein = 810 episodic steps
- Scalarized REINFORCE: K_rein = 810 episodic steps

## Final f(J) — mean ± SD across 20 seeds (all algos)

| α | MLMC | Bai-PG | REIN (paper A) | Scal REIN | M−Bai (p) | M−pA (p) | M−Scal (p) |
|---|---|---|---|---|---|---|---|
| 0.10 | **43.27 ± 2.01** | 40.83 ± 2.00 | 37.40 ± 4.64 | 40.61 ± 2.46 | +2.43 (**0.0001**) | +5.87 (<0.0001) | +2.66 (0.0009) |
| 0.25 | **38.25 ± 1.80** | 36.26 ± 1.90 | 33.47 ± 3.55 | 36.56 ± 1.42 | +1.98 (**0.0013**) | +4.78 (<0.0001) | +1.69 (0.0016) |
| 0.50 | **33.87 ± 1.00** | 32.93 ± 1.17 | 30.94 ± 2.28 | 33.01 ± 0.94 | +0.94 (**0.0026**) | +2.93 (<0.0001) | +0.86 (0.0103) |
| 1.00 | **12.41 ± 0.35** | 12.11 ± 0.45 | 11.19 ± 0.95 | 11.80 ± 0.74 | +0.30 (**0.0087**) | +1.22 (<0.0001) | +0.61 (0.0001) |

p-values are Mann-Whitney U exact two-sided. MLMC-NPG wins all four α
against all three alternatives at p ≤ 0.01.

## Files in this directory

```
ft_nn32_B16_NK40_a*_fair_budget.png         learning-curve plots, 4 algos
ft_nn32_B16_NK40_a*_fair_budget.tikz        TikZ source (for LaTeX)
ft_nn32_B16_NK40_a*_plateaus.csv            per-algo plateau mean/std
ft_nn32_B16_NK40_a*_pvalues.txt             MWU vs MLMC
README.md                                   this file
```

Raw per-seed JSON (with full f(J) trajectory, J_hat trajectory, NN
parameters at the end, and config) lives in
`results/sweep_v20_pre_final_ft/`.
