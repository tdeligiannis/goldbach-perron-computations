# Experiment 5: Bulk Cancellation, KMT Variance, and Exponential Sums

**Paper reference:** Appendix A, §A.5

Three sub-experiments testing equidistribution of g_k(n) = λ(n)·λ(n+k)
in arithmetic progressions.

## 5a: Bulk cancellation (`bulk_cancellation.py`)

Computes |Σ_{n≤x} λ(n)·λ(n+h)| / x for shifts h ∈ {1,3,7,11,13,17,19,23,29,37}
(odd primes, bias-corrected).  Fits decay exponents α in |Σ|/x ~ (ln x)^{−α}
and compares polynomial vs. subexponential models.

Includes random CM comparison (seed 42) to distinguish λ-specific effects
from generic CM function behaviour.

```bash
# Edit X_MAX at top of script, then:
python3 bulk_cancellation.py
```

Full scale (X = 10¹⁰) requires ~80 GB per worker for the cumulative sum array.

## 5b: KMT variance (`kmt_variance.py`)

Tests whether g_k satisfies a KMT-type variance bound in arithmetic
progressions mod q ≤ 500.  Computes V(X,Q)·Q/X² (normalized cumulative
variance) and max|ĝ(a/q)|/X (exponential sum peaks).

```bash
# Edit X and Q_MAX at top of script, then:
python3 kmt_variance.py
```

## 5c: Multi-scale exponential sums (`multiscale_expsums.py`)

Computes max_{(a,q)=1} |ĝ(a/q)|/X at scales X ∈ {10⁵, …, 10⁸} for
fixed moduli q.  Tests whether the decay rate matches the CLT prediction
X^{−1/2} or exceeds it.

```bash
python3 multiscale_expsums.py --test           # quick test
python3 multiscale_expsums.py --xmax 8         # X up to 10^8
```

## Key results

| Test | Observed | Goldbach needs |
|------|----------|----------------|
| Bulk α (corrected) | 4.63 ± 0.78 | ≥ 2 |
| V_ratio (max deviation) | 3.5% | O(1) |
| \|ĝ\|/X at X = 10⁸ | 3.4 × 10⁻⁴ | ≤ 0.041 |
| \|ĝ\|/X decay rate | X^{−0.49} | ≥ (log X)^{−1.1} |
