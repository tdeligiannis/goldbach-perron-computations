# Experiment 2: Turán–Kubilius Two-Prime Interactions

**Paper reference:** Appendix A, §A.2

## What it computes

For 58 primes p ∈ [11, 300] at X = 2 × 10⁹, computes the two-prime
TK interaction:

  A_{7,p} = (1/X) Σ_{n≤X} Δ₇(n) · Δ_p(n)

where Δ_p(n) = C_p(n) − E_p is the mean-zero fluctuation of
(−1)^{v_p(Q_k(n))} with Q_k(n) = n(n+k)(n+h)(n+k+h), k = 4, h = 6.

Also computes the coherent partial sum C = Σ_p coeff_p · A_{7,p}
vs. the incoherent bound I = Σ_p |coeff_p| · |A_{7,p}|, and the
cancellation ratio r = |C|/I.

## Key result

Observed median |A_{7,p}| ≈ 1.39 × 10⁻⁸, which is:
- 1,600× below random-walk (GRH-level) bound X^{−1/2}
- 5.7 × 10⁴× below the CRT–truncation bound X^{−1/3}

The empirical scaling |A_{7,p}| ~ X^{−0.845} implies sub-polynomial
cancellation, far beyond any currently provable rate.

## Building and running

```bash
# Compile
gcc -O3 -march=native -fopenmp -o tk_two_prime tk_two_prime.c -lm

# Quick test (X = 5×10⁷, ~2 min on 4 cores)
gcc -O3 -march=native -fopenmp -DX_MAX=50000000ULL -o tk_test tk_two_prime.c -lm
OMP_NUM_THREADS=4 ./tk_test

# Full run (X = 2×10⁹, ~4h on 64 cores, ~4 GB RAM)
OMP_NUM_THREADS=64 ./tk_two_prime 2>&1 | tee results_tk.txt
```

## Memory requirements

| X_MAX | RAM |
|-------|-----|
| 5×10⁷ | ~100 MB |
| 2×10⁹ | ~4 GB |
| 10¹⁰ | ~20 GB |

## Output

Per-prime table of A_{7,p}, sign, coherent/incoherent partial sums,
and cancellation ratio.  Matches the paper's Table 2 (§A.2).

## What to look for

- **Cancellation ratio r → 0** as more primes are added: coherent
  cancellation in the TK sum.
- **Sign pattern**: ~50% negative A_{7,p} (mixed signs enable
  cancellation).  At X = 2×10⁹, fraction is 0.724 negative
  (statistically significant skew, binomial p-value 0.0009).
- **r decreasing with X**: genuine (slow) cancellation mechanism.
