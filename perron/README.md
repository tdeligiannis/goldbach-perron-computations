# Experiment 3: Four-Scale W^{2,2} Sobolev Regularity Verification

**Paper reference:** Appendix A, §A.3 and Part III (§§12–15)

## What it computes

For M ∈ {10⁷, 10⁸, 10⁹, 10¹⁰}, computes the Dirichlet series
G_a(s) = Σ_{m≥1} g_a(m)/m^s and its derivative G_a'(s) at s = 1 + it
for 15 values of t from 0.5 down to 10⁻⁵, where
g_a(m) = λ(Wm + a) · λ(Wm + a + h) with W = 30, a = 1, h = 6.

Three observables:
1. **|G_a'(1+it)|** — should converge to a finite limit ≈ 1.264
   (W^{2,2} Sobolev regularity ⟹ C¹ by Sobolev embedding)
2. **|H(t)|/t = |G_a(1+it) − G_a(1)|/t** — should converge to
   |G_a'(1)| ≈ 1.264 (Lipschitz, confirming W^{2,2} not just W^{1,2})
3. **Cesàro ratio |S(M)/M| · (log M)²** — should stay below 1
   (the Goldbach threshold c ≥ 2)

## Key result

At M = 10¹⁰: |G_a'(1)| = 1.2642 (drift decelerating geometrically),
Cesàro ratio = 0.009 (114× below Goldbach threshold).
|H(t)|/t → 1.2642 while |H(t)|/√t → 0, ruling out Hölder-1/2.

## Running

```bash
# Quick test (M = 10⁷, ~1 min)
python3 goldbach_perron_computation.py --mmax 10000000 --cores 4

# Full run (M = 10¹⁰, ~2h on 64 cores)
python3 goldbach_perron_computation.py --mmax 10000000000 --cores 64 --outdir results

# Regenerate plots from saved data
python3 replot.py --datadir results --outdir plots
```

## Files

| File | Description |
|---|---|
| `goldbach_perron_computation.py` | Main computation script |
| `replot.py` | Regenerate plots from `perron_data.npz` |
| `results/perron_data.npz` | Saved numerical data (all scales) |
| `plots/plot1_Ga_prime_bounded.png` | |G_a'(1+it)| vs t |
| `plots/plot2_lipschitz_test.png` | Lipschitz test |H(t)|/t |
| `plots/plot3_cesaro_ratio.png` | Cesàro ratio vs Goldbach threshold |

## Memory

~30 GB at M = 10¹⁰ (segmented, parallelised across cores).
