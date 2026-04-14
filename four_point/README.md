# Experiment 1: Four-Point Covariance Splitting

**Paper reference:** Appendix A, §A.1

## What it computes

For shifts k ∈ {11, 13, 17, 19, 23, 29, 31} (odd primes with gcd(k, h) = 1, h = 6)
at checkpoints X ∈ {10⁷, …, 10¹⁰}:

- C₄(k) = (1/X) Σ_{n≤X} λ(n)λ(n+k)λ(n+h)λ(n+k+h)
- μ₁ = (1/X) Σ λ(n)λ(n+k)
- μ₂ = (1/X) Σ λ(n+h)λ(n+k+h)
- Cov = C₄ − μ₁μ₂
- Decay exponents α from |C₄| ~ (log X)^{−α}

## Key result

At X = 3 × 10⁹: |C₄(k)| is 97–2625× below the Goldbach threshold (log X)^{−2}.
Fitted α ≈ 8–18, far exceeding the required α ≥ 2.

## Building and running

```bash
# Compile
gcc -O3 -march=native -fopenmp -o c4_covariance c4_covariance.c -lm

# Quick test (X = 5×10⁷, ~1 min on 4 cores)
OMP_NUM_THREADS=4 ./c4_covariance   # with -DX_MAX=50000000ULL at compile

# Full run (X = 10¹⁰, ~8h on 64 cores, ~11 GB RAM)
OMP_NUM_THREADS=64 ./c4_covariance 2>&1 | tee results_c4.txt
```

To change X_MAX, either edit the `#define` or pass it at compile time:
```bash
gcc -O3 -march=native -fopenmp -DX_MAX=3000000000ULL -o c4_covariance c4_covariance.c -lm
```

## Memory requirements

| X_MAX | RAM |
|-------|-----|
| 5×10⁷ | ~50 MB |
| 10⁹ | ~1 GB |
| 10¹⁰ | ~11 GB |

## Output

Produces formatted tables matching the paper's Table 1 (§A.1), including
C₄(k), μ₁, μ₂, |Cov|, and fitted decay exponents per shift k.

## What to look for

- **α_C4 ≥ 2**: sufficient for Goldbach (via Bridge Lemma).
- **|Cov|/|C₄| ≈ 1**: confirms μ₁μ₂ factorisation is irrelevant.
- **Consistency across k**: decay is a four-point structural property,
  not k-specific.
