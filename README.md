# Goldbach–Perron Computational Verification

Computational evidence for the smoothed Perron bound on two-point Liouville correlations, supporting the paper sequence:

- **Flagship:** *Polynomial correlations of the Liouville function with an application to an additive problem* ([Zenodo DOI: 10.5281/zenodo.19474787](https://doi.org/10.5281/zenodo.19474787))
- **Perron note:** *A smoothed Perron bound for two-point Liouville correlations via Sobolev regularity at σ=1* (Zenodo, April 2026)
- **Unified paper:** *Polynomial correlations of the Liouville function and the binary Goldbach conjecture* (in preparation)

## What this verifies

The smoothed Perron argument claims that the two-point Liouville correlation satisfies S_Φ(M) = O(M/(log M)²) unconditionally, using three inputs: Tao's continuity theorem, the Montgomery–Vaughan L² mean-value theorem, and the Sobolev embedding H² ↪ C¹ in one dimension. The code verifies every quantitative prediction across four decades (M = 10⁷, 10⁸, 10⁹, 10¹⁰):

| Quantity | Prediction | Observed |
|---|---|---|
| \|G_a'(1+it)\| | Bounded (constant in t) | 1.264, stable to 6 digits |
| \|H(t)\|/t | Converges to \|G_a'(1)\| (Lipschitz) | 1.2642, stable from t = 0.002 to 10⁻⁵ |
| \|S/M\| · (log M)² | Below 1 (Goldbach threshold) | 0.009 at M = 10¹⁰ (114× below) |
| G_a(1) | Finite nonzero constant | 0.526, stable to 3 digits |

## Files

| File | Description |
|---|---|
| `goldbach_perron_computation.py` | Main computation: segmented Liouville sieve, parallel evaluation of G_a(1+it) and G_a'(1+it) at 15 values of t across multiple decades of M, with publication-quality PNG plots |
| `replot.py` | Regenerate plots from saved data without rerunning the sieve |
| `perron_data.npz` | Saved numerical results from the 4-decade run (load with `numpy`) |
| `plot1_Ga_prime_bounded.png` | Boundedness of \|G_a'(1+it)\| at σ = 1 |
| `plot2_lipschitz_test.png` | Lipschitz test: \|H(t)\|/t → \|G_a'(1)\| |
| `plot3_cesaro_ratio.png` | Cesàro ratio vs Goldbach threshold |

## Usage

### Requirements

```
pip install numpy matplotlib
```

### Quick test (M = 10⁷, ~10 seconds)

```bash
python3 goldbach_perron_computation.py --mmax 10000000 --cores 4 --seg 500000 --outdir results
```

### Full 4-decade run (M = 10¹⁰, ~45 minutes on 64 cores)

```bash
python3 goldbach_perron_computation.py --mmax 10000000000 --cores 64 --seg 5000000 --outdir results
```

### Regenerate plots from saved data

```bash
python3 replot.py
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--mmax` | 10⁹ | Maximum value of M |
| `--cores` | 64 | Number of CPU cores for parallel sieve |
| `--seg` | 5×10⁶ | Segment size in m-values per worker |
| `--outdir` | `.` | Output directory for plots and data |

## Method

The code computes g_a(m) = λ(30m+1)·λ(30m+7) for m ≤ M using a segmented sieve of Ω(n) mod 2 (the Liouville function). For each M, it evaluates:

- G_a(1+it; M) = Σ_{m≤M} g_a(m) / m^{1+it}
- G_a'(1+it; M) = -Σ_{m≤M} g_a(m) (log m) / m^{1+it}

at 15 values of t from 0.5 down to 10⁻⁵. The sieve is parallelized across segments using Python's `multiprocessing.Pool`.

## Hardware

The full 4-decade run was performed on a 64-core workstation with 540 GB RAM running Pop!_OS Linux. The computation at M = 10¹⁰ (corresponding to n ≤ 3×10¹¹) required approximately 45 minutes.

## License

MIT

## Author

Theodore Deligiannis, Multiscale Lab, University of Nebraska at Omaha

## Citation

If you use this code, please cite:

```
Deligiannis, T. (2026). Polynomial correlations of the Liouville function
with an application to an additive problem. Zenodo.
https://doi.org/10.5281/zenodo.19474787
```
