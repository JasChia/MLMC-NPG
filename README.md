# pre-final Fruit Tree suite — v16 K=40 sweep at NN-32 B=16 across α

The four PNGs in this folder are fair-budget curves on Fruit Tree (depth=7) NN-32
with B=16, N=20, K=40, β=0.05, lr=0.05, γ=0.99, δ=0.5, comparing
**MLMC-NPG vs Bai-PG vs paper-A REINFORCE vs scalarized REINFORCE**, varying only α.

| plot | α | regime |
|---|---|---|
| `ft_nn32_B16_NK40_a010_fair_budget.png` | 0.1 | near-linear α-fairness |
| `ft_nn32_B16_NK40_a025_fair_budget.png` | 0.25 | mild concavity |
| `ft_nn32_B16_NK40_a050_fair_budget.png` | 0.5 | canonical 2·√ fairness |
| `ft_nn32_B16_NK40_a100_fair_budget.png` | 1.0 | log fairness |

## Reproduce

```bash
pip install numpy torch scipy matplotlib

# Run the four sweeps (3 seeds, 4 algos, ~6,500 trajectories per algo per sweep):
for ALPHA in 0.1 0.25 0.5 1.0; do
  TAG="ft_nn32_B16_NK40_a$(printf '%03d' $(python -c "print(int(round($ALPHA*100)))"))"
  python -m alt.run_v15_search \
    --out_dir results/sweep_v16 --tag "$TAG" \
    --env_kind fruit_tree --alpha "$ALPHA" \
    --policy nn --hidden 32 \
    --B 16 --N 20 --K 40 --delta 0.5 \
    --seeds 7 --workers 2
done

# Render the four plots:
for ALPHA in 010 025 050 100; do
  python -m alt.plot_v14 \
    --results_dir results/sweep_v16 \
    --tag "ft_nn32_B16_NK40_a${ALPHA}" \
    --out_dir pre_final_suite_fruit_tree \
    --title "Fruit Tree NN-32 alpha=$(echo $ALPHA | sed 's/^0*//;s/^/0./;s/^0\.$/0/') B=16 N=20 K=40"
done
```

The runner auto-derives `K_pg` for Bai-PG and `K_rein` for both REINFORCE
variants so all four algorithms consume the same total trajectory budget
(≈ 6,490 trajectories each at this config). See
[`alt/run_v15_search.py`](../alt/run_v15_search.py)::`derive_K`.

## Single-plot example

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a010 \
  --env_kind fruit_tree --alpha 0.1 \
  --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 \
  --seeds 7 --workers 2

python -m alt.plot_v14 \
  --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a010 \
  --out_dir pre_final_suite_fruit_tree \
  --title "Fruit Tree NN-32 alpha=0.1 B=16 N=20 K=40 [7 seeds]"
```

Each `plot_v14` invocation also writes `<tag>_plateaus.csv` (per-algo last-K/10
mean and sd) and `<tag>_pvalues.txt` (Wilcoxon rank-sum MLMC vs each
alternative) into the same output directory.
