# Experiment 4: Wavelet Verification of Pointwise Cancellation

**Paper reference:** Appendix A, §A.4

## What it computes

Haar wavelet coefficients C_j(k, h) of g(n) = λ(n)·λ(n+h) at
dyadic scales j = 16..60, testing the **pointwise bound**:

    max_k |C_j(k,h)| / 2^j  ≤  1/j²

If this holds for all j ≥ j₀, the wavelet-based Sobolev regularity
argument yields effective c_MR = β/2 where MS(j) ~ j^{−β}.

## Key result

- N = 4 × 10¹¹, h = 2, 12,206,994 blocks, 64 threads
- Pointwise bound holds for all j ≥ 23 (15 consecutive scales, zero exceptions)
- Transition at j = 22→23 (ratio 1.04 → 0.76)
- MS power-law fit: MS(j) ~ j^{−8.70}, giving effective c_MR = 4.35

## Building and running (C — primary)

```bash
gcc -O3 -march=native -fopenmp -o goldbach_wavelet goldbach_wavelet.c -lm

./goldbach_wavelet 10      # N=10B, ~10 GB, ~5 min
./goldbach_wavelet 50      # N=50B, ~50 GB, ~20 min
./goldbach_wavelet 200     # N=200B, ~200 GB, ~2 hrs
./goldbach_wavelet 400     # N=400B, ~400 GB, ~4 hrs
./goldbach_wavelet 10 6    # custom shift h=6
```

## Python alternative (small-scale verification)

```bash
python3 wavelet_verification.py --test                    # ~10s
python3 wavelet_verification.py --nmax 100000000          # N=10^8, ~5 min
```

## Memory

N+1 bytes (one int8 per integer). N = 400B requires ~400 GB.
