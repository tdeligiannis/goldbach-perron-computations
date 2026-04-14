# Goldbach–Perron Computations

Computational experiments accompanying the paper:

> T. Deligiannis, **"A reduction of binary Goldbach to a four-point Chowla bound
> via the polynomial squaring identity"** (2026).
> DOI: [10.5281/zenodo.19474787](https://doi.org/10.5281/zenodo.19474787)

Five independent experiments confirm every quantitative prediction of the paper
across four decades of computation ($X$ up to $10^{10}$, $N$ up to $4 \times 10^{11}$).

## Experiments at a glance

| # | Experiment | Paper §  | Language | Key observable | Result |
|---|-----------|----------|----------|----------------|--------|
| 1 | Four-point covariance splitting | §A.1 | C/OpenMP | $\|C_4(k)\| \lesssim (\log X)^{-\alpha}$ | $\alpha \approx 8\text{--}18$ (threshold: $\alpha \geq 2$) |
| 2 | Turán–Kubilius two-prime interactions | §A.2 | C/OpenMP | $\|A_{7,p}\|$ | $1.39 \times 10^{-8}$, 1600× below $X^{-1/2}$ |
| 3 | $W^{2,2}$ Sobolev regularity | §A.3 | Python | $\|G_a'(1+it)\|$ | Converges to 1.2642 (bounded) |
| 4 | Wavelet verification | §A.4 | C/OpenMP | $\max\|C_j\|/2^j \leq 1/j^2$ | Holds for all $j \geq 23$ |
| 5 | Bulk cancellation & exponential sums | §A.5 | Python | Bulk $\alpha$, $\|\hat{g}\|/X$ | $\alpha = 4.63 \pm 0.78$ |

## Repository structure

```
├── README.md                          ← this file
├── LICENSE                            ← MIT
├── Makefile                           ← build and run all experiments
├── common/
│   └── liouville_sieve.py             ← shared segmented Liouville sieve
├── four_point/
│   ├── c4_covariance.c                ← Experiment 1 (C/OpenMP)
│   └── README.md
├── turan_kubilius/
│   ├── tk_two_prime.c                 ← Experiment 2 (C/OpenMP)
│   └── README.md
├── perron/
│   ├── goldbach_perron_computation.py ← Experiment 3
│   ├── replot.py                      ← regenerate plots from saved data
│   ├── results/perron_data.npz        ← saved numerical results
│   ├── plots/                         ← publication-quality PNG figures
│   └── README.md
├── wavelet/
│   ├── goldbach_wavelet.c             ← Experiment 4 (C/OpenMP, primary)
│   ├── wavelet_verification.py        ← Experiment 4 (Python, small-scale)
│   └── README.md
└── exponential_sums/
    ├── bulk_cancellation.py           ← Experiment 5a: bulk Σλ(n)λ(n+h)
    ├── kmt_variance.py                ← Experiment 5b: KMT variance
    ├── multiscale_expsums.py          ← Experiment 5c: |ĝ(a/q)|/X decay
    └── README.md
```

## Quick start

### Prerequisites

```bash
# C programs (Linux)
sudo apt install build-essential       # gcc
# Python
pip install numpy matplotlib scipy     # Python 3.8+
```

### Compile everything

```bash
make all
```

### Smoke tests (~2 minutes, any machine)

```bash
make test
```

This runs each C program at small scale ($X = 5 \times 10^7$) and the
Python experiments in test mode.

### Individual quick tests

```bash
# Experiment 1: C4 covariance (X=50M, ~30s)
gcc -O3 -march=native -fopenmp -DX_MAX=50000000ULL \
    -o test_c4 four_point/c4_covariance.c -lm
OMP_NUM_THREADS=4 ./test_c4

# Experiment 2: TK interactions (X=50M, ~30s)
gcc -O3 -march=native -fopenmp -DX_MAX=50000000ULL \
    -o test_tk turan_kubilius/tk_two_prime.c -lm
OMP_NUM_THREADS=4 ./test_tk

# Experiment 3: Perron regularity (M=10^7, ~1 min)
cd perron && python3 goldbach_perron_computation.py --mmax 10000000 --cores 4

# Experiment 4: Wavelet (N=1B, ~1 min)
cd wavelet && ./goldbach_wavelet 1

# Experiment 5c: Exponential sums (X=10^5, ~5s)
cd exponential_sums && python3 multiscale_expsums.py --test
```

### Production runs (64-core workstation)

```bash
make run-exp1    # Exp 1: C4, X=10^10, ~8h, 11 GB
make run-exp2    # Exp 2: TK, X=2×10^9, ~4h, 4 GB
make run-exp3    # Exp 3: Perron, M=10^10, ~2h, 30 GB
make run-exp4    # Exp 4: Wavelet, N=10B, ~5 min, 10 GB
make run-exp5    # Exp 5c: Expsums, X=10^8, ~15 min, 2 GB
```

## Paper cross-reference

| Paper table/figure | Experiment | Script |
|---|---|---|
| Table A.1: $\|C_4(k)\|$ vs $(\log X)^{-2}$ | 1 | `four_point/c4_covariance.c` |
| Table A.2: $\|A_{7,p}\|$ benchmarks | 2 | `turan_kubilius/tk_two_prime.c` |
| Table A.3: $\|G_a'(1)\|$ convergence | 3 | `perron/goldbach_perron_computation.py` |
| Table A.4: Lipschitz test | 3 | `perron/goldbach_perron_computation.py` |
| Table A.5: Cesàro ratio | 3 | `perron/goldbach_perron_computation.py` |
| Table A.6: Wavelet scales | 4 | `wavelet/goldbach_wavelet.c` |
| Table A.7: Multi-scale $\|\hat{g}\|/X$ | 5c | `exponential_sums/multiscale_expsums.py` |
| Table A.8: Summary of all tests | all | — |
| Figure 1: $\|G_a'(1+it)\|$ | 3 | `perron/plots/plot1_Ga_prime_bounded.png` |
| Figure 2: Lipschitz test | 3 | `perron/plots/plot2_lipschitz_test.png` |
| Figure 3: Cesàro ratio | 3 | `perron/plots/plot3_cesaro_ratio.png` |
| Figure 5: KMT normalized variance | 5b | `exponential_sums/kmt_variance.py` |

## Hardware requirements

All full-scale computations were performed on a single workstation:
**64-core CPU, 540 GB RAM, Pop!\_OS Linux**.

| Experiment | X/N | RAM | Cores | Time |
|---|---|---|---|---|
| 1: C4 covariance | $10^{10}$ | 11 GB | 64 | ~8h |
| 2: TK interactions | $2 \times 10^9$ | 4 GB | 64 | ~4h |
| 3: Perron | $10^{10}$ | 30 GB | 64 | ~2h |
| 4: Wavelet | $4 \times 10^{11}$ | 400 GB | 64 | ~4h |
| 5a: Bulk cancellation | $10^{10}$ | 80 GB/worker | 6 | ~2h |
| 5b: KMT variance | $10^8$ | 2 GB | 24 | ~20 min |
| 5c: Exponential sums | $10^8$ | 2 GB | 1 | ~15 min |

All scripts have test modes that run on any machine (4+ cores, 4+ GB RAM)
in under 30 seconds.

## Known numerical artefact

At $X = 10^{10}$, all seven shifts simultaneously exhibit $\mu_1 \approx -0.00280$
(coefficient of variation 2.4%).  This is a genuine long-range Liouville oscillation
— a Chebyshev-type bias associated with zeros of $\zeta$ near the real axis —
confirmed not to be a computational artefact.  All exponent regressions use
$X \leq 3 \times 10^9$ as the reliable asymptotic range.

## License

MIT — see [LICENSE](LICENSE).
