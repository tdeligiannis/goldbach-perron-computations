#!/usr/bin/env python3
"""
Experiment 5b: KMT Variance Bound Test for Binary Liouville Correlations
==========================================================================

Paper: T. Deligiannis, "A reduction of binary Goldbach to a
       four-point Chowla bound via the polynomial squaring identity"
       (2026), Appendix A, §A.5 (KMT variance).

Tests whether g(n) = λ(n)λ(n+k) satisfies a KMT-type variance bound
in arithmetic progressions, and whether |ĝ(a/q)|/X decays to 0 with X.

WHAT WE'RE LOOKING FOR:
  1. R(Q) = V(X,Q)·Q/X² should be comparable for g_k and baseline λ
  2. |ĝ(a/q)|/X should be comparable between g_k and λ 
  3. No anomalous moduli q where g_k deviates from baseline

If confirmed at X=10^8: input (c) of the circle method is supported,
    implying C₄(k) = O((ln X)^{-A}) for any A → Goldbach.

USAGE: Run in Spyder or from command line:
    python kmt_variance_test.py

Estimated runtime: 15-25 min at X=10^8 on a 24-core system.
Memory: ~2 GB.
"""

import numpy as np
from math import gcd, isqrt, log
import time
import json
import os

# ============================================================
# PARAMETERS — adjust for your system
# ============================================================
X        = 10**8       # Main range (10^8 target; use 10^6 for quick test)
Q_MAX    = 500         # Maximum modulus for variance test
SHIFTS   = [1, 2, 3, 6]
SAVE_DIR = os.path.expanduser("~/kmt_results")

# For multi-scale analysis: also run at smaller X to see scaling
MULTI_SCALE = True
SCALES   = [10**5, 10**6, 10**7, 10**8]  # X values to test

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SIEVE FOR λ(n)
# ============================================================
def compute_liouville(N):
    """
    Compute λ(n) = (-1)^Ω(n) for n = 0, 1, ..., N.
    Returns int8 array with λ(0) = 0.
    """
    print(f"\n{'='*60}")
    print(f" SIEVING λ(n) for n ≤ {N:,}")
    print(f"{'='*60}")
    t0 = time.time()

    lam = np.ones(N + 1, dtype=np.int8)
    lam[0] = 0

    sqrt_N = isqrt(N)

    # Phase 1: Boolean sieve of Eratosthenes
    print(f"  Phase 1: Sieve of Eratosthenes...")
    t1 = time.time()
    sieve = np.ones(N + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, sqrt_N + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    print(f"    Done in {time.time()-t1:.1f}s")

    # Phase 2: Small primes (≤ √N) and their powers → flip λ
    small_primes = np.where(sieve[:sqrt_N + 1])[0]
    print(f"  Phase 2: {len(small_primes):,} small primes...")
    t1 = time.time()
    for p in small_primes:
        pk = int(p)
        while pk <= N:
            lam[pk::pk] *= -1
            pk *= int(p)
    print(f"    Done in {time.time()-t1:.1f}s")

    # Phase 3: Large primes (> √N) → each has ≤ √N multiples
    large_primes = np.where(sieve[sqrt_N + 1:])[0] + sqrt_N + 1
    n_large = len(large_primes)
    print(f"  Phase 3: {n_large:,} large primes...")
    t1 = time.time()

    # Batch: primes > N/2 have only themselves as a multiple ≤ N
    cutoff = N // 2
    big = large_primes[large_primes > cutoff]
    lam[big] *= -1
    print(f"    {len(big):,} primes > N/2 (batch flip)")

    # Remaining large primes: √N < p ≤ N/2
    remaining = large_primes[large_primes <= cutoff]
    print(f"    {len(remaining):,} primes in (√N, N/2]...", flush=True)
    t2 = time.time()
    for idx, p in enumerate(remaining):
        lam[p::p] *= -1
        if idx % 500000 == 0 and idx > 0:
            print(f"      {idx:,}/{len(remaining):,}  ({time.time()-t2:.1f}s)", flush=True)
    print(f"    Done in {time.time()-t1:.1f}s")

    # Verification
    checks = {1:1, 2:-1, 3:-1, 4:1, 5:-1, 6:1, 7:-1, 8:-1,
              9:1, 10:1, 12:-1, 16:1, 30:-1, 36:1, 64:1, 100:1}
    for n, v in checks.items():
        if n <= N:
            assert lam[n] == v, f"FAIL: λ({n})={lam[n]}, expected {v}"

    L_1k = int(np.sum(lam[1:1001].astype(np.int64)))
    print(f"  Verification ✓  L(1000)={L_1k}")
    print(f"  Total sieve: {time.time()-t0:.1f}s, "
          f"Memory: {lam.nbytes/1e6:.0f} MB")

    del sieve
    return lam


# ============================================================
# RESIDUE CLASS SUMS
# ============================================================
def residue_sums(g, q):
    """
    For g[0..N-1] representing g(1), g(2), ..., g(N),
    compute S[r] = Σ_{n ≡ r (mod q), 1≤n≤N} g(n) for r = 0, ..., q-1.
    """
    N = len(g)
    full = (N // q) * q

    # Column j of reshaped array collects g(n) for n ≡ (j+1) mod q
    col_sums = g[:full].reshape(-1, q).sum(axis=0, dtype=np.int64)
    S = np.roll(col_sums, 1)  # S[r] = col_sums[(r-1) % q]

    # Handle remainder
    for i in range(full, N):
        S[(i + 1) % q] += int(g[i])

    return S


def coprime_mask_q(q):
    """Boolean mask: mask[a] = True iff gcd(a,q) = 1."""
    mask = np.ones(q, dtype=bool)
    mask[0] = False
    temp = q
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            mask[::p] = False
            while temp % p == 0:
                temp //= p
        p += 1
    if temp > 1:
        mask[::temp] = False
    return mask


# ============================================================
# PER-MODULUS ANALYSIS
# ============================================================
def analyze_modulus(g, q, k=None):
    """Compute variance, sup-norm, and exponential sum for modulus q."""
    X_eff = len(g)
    S = residue_sums(g, q)
    mask = coprime_mask_q(q)
    phi_q = int(mask.sum())
    if phi_q == 0:
        return None

    S_coprime = S[mask]
    mean_val = S_coprime.sum() / phi_q

    V_q = float(np.sum((S_coprime.astype(np.float64) - mean_val)**2))
    sup_S = float(np.max(np.abs(S_coprime)))
    sup_norm = sup_S / (X_eff / q)

    # Exponential sum via DFT: ĝ(a/q) = Σ_r S[r] e(ra/q)
    dft = np.fft.fft(S.astype(np.complex128))
    mags = np.abs(dft)
    coprime_a = np.where(mask)[0]
    exp_max = float(max(mags[a] for a in coprime_a)) if len(coprime_a) > 0 else 0.0
    exp_mean = float(np.mean([mags[a] for a in coprime_a])) if len(coprime_a) > 0 else 0.0

    result = {
        'q': int(q),
        'phi_q': phi_q,
        'V_q': V_q,
        'sup_norm': sup_norm,
        'sup_S': sup_S,
        'mean_coprime': float(mean_val),
        'exp_sum_max': exp_max,
        'exp_sum_max_normed': exp_max / X_eff,
        'exp_sum_mean_normed': exp_mean / X_eff,
    }
    if k is not None:
        result['coprime_to_k'] = (gcd(q, k) == 1)
    return result


# ============================================================
# VARIANCE SWEEP
# ============================================================
def variance_sweep(g, Q_max, label, k=None):
    """Sweep q = 2 to Q_max, returning per-q and cumulative results."""
    print(f"\n  Variance sweep: {label}, Q_max={Q_max}")
    t0 = time.time()
    X_eff = len(g)

    results = []
    V_cum = 0.0

    for q in range(2, Q_max + 1):
        r = analyze_modulus(g, q, k=k)
        if r is None:
            continue
        V_cum += r['V_q']
        r['V_cumul'] = V_cum
        r['R_Q'] = V_cum * q / X_eff**2
        results.append(r)

        if q % 100 == 0:
            print(f"    q={q:4d}  V_cum={V_cum:.3e}  R(Q)={r['R_Q']:.6f}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"    Complete ({time.time()-t0:.1f}s)")
    return results


# ============================================================
# MULTI-SCALE ANALYSIS (the key scaling test)
# ============================================================
def multi_scale_analysis(lam, scales, shifts, q_test_values):
    """
    For each X in scales, compute |ĝ(a/q)|/X at fixed q values.
    This reveals how the exponential sum decays with X.
    """
    print(f"\n{'='*60}")
    print(f" MULTI-SCALE ANALYSIS: X ∈ {scales}")
    print(f"{'='*60}")

    results = {}  # results[k][X_val] = {q: max |ĝ(a/q)|/X}

    for k in shifts:
        results[k] = {}
        for X_val in scales:
            if X_val + k >= len(lam):
                continue
            g = (lam[1:X_val+1] * lam[1+k:X_val+1+k]).astype(np.int8)
            exp_data = {}
            for q in q_test_values:
                if q >= X_val:
                    continue
                S = residue_sums(g, q)
                dft = np.fft.fft(S.astype(np.complex128))
                mags = np.abs(dft)
                mask = coprime_mask_q(q)
                coprime_a = np.where(mask)[0]
                if len(coprime_a) > 0:
                    exp_data[q] = float(max(mags[a] for a in coprime_a)) / X_val
            results[k][X_val] = exp_data

    # Also do baseline (multiplicative λ)
    results['baseline'] = {}
    for X_val in scales:
        g = lam[1:X_val+1].copy()
        exp_data = {}
        for q in q_test_values:
            if q >= X_val:
                continue
            S = residue_sums(g, q)
            dft = np.fft.fft(S.astype(np.complex128))
            mags = np.abs(dft)
            mask = coprime_mask_q(q)
            coprime_a = np.where(mask)[0]
            if len(coprime_a) > 0:
                exp_data[q] = float(max(mags[a] for a in coprime_a)) / X_val
        results['baseline'][X_val] = exp_data

    # Print table
    print(f"\n  max |ĝ(a/q)|/X  (should decrease with X if KMT-type bound holds)")
    print(f"  {'':8s}", end='')
    for X_val in scales:
        print(f"  X={X_val:.0e}", end='')
    print(f"  1/(lnX)^1.1")
    print(f"  {'-'*80}")

    for q in q_test_values:
        # Baseline
        print(f"  q={q:<3d} λ ", end='')
        for X_val in scales:
            v = results['baseline'].get(X_val, {}).get(q, float('nan'))
            print(f"  {v:>9.6f}", end='')
        ref = 1 / log(scales[-1])**1.1
        print(f"  {ref:.6f}")

        for k in shifts:
            print(f"       k={k}", end='')
            for X_val in scales:
                v = results.get(k, {}).get(X_val, {}).get(q, float('nan'))
                print(f"  {v:>9.6f}", end='')
            print()
        print()

    return results


# ============================================================
# PLOTS
# ============================================================
def generate_plots(all_results, baseline_results, X_eff, save_dir,
                   multi_scale_data=None):
    """Generate all diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n  Generating plots → {save_dir}")
    colors = {'baseline': 'black', 1: '#e41a1c', 2: '#377eb8',
              3: '#4daf4a', 6: '#984ea3'}

    # --- Plot 1: R(Q) = V(X,Q)·Q/X² ---
    fig, ax = plt.subplots(figsize=(10, 7))
    qs = [r['q'] for r in baseline_results]
    Rs = [r['R_Q'] for r in baseline_results]
    ax.plot(qs, Rs, 'k-', lw=2, label='λ(n) (multiplicative)', alpha=0.8)

    for k, results in all_results.items():
        qs = [r['q'] for r in results]
        Rs = [r['R_Q'] for r in results]
        ax.plot(qs, Rs, '-', color=colors[k], lw=1.5,
                label=f'λ(n)λ(n+{k})', alpha=0.8)

    ax.set_xlabel('Q (max modulus)', fontsize=13)
    ax.set_ylabel('R(Q) = V(X,Q) · Q / X²', fontsize=13)
    ax.set_title(f'KMT Normalized Variance — X = {X_eff:,}\n'
                 f'Similar curves ⟹ g_k equidistributes like multiplicative f',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plot1_R_Q.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Per-q exponential sum max ---
    fig, ax = plt.subplots(figsize=(10, 7))
    qs_b = [r['q'] for r in baseline_results]
    es_b = [r['exp_sum_max_normed'] for r in baseline_results]
    ax.semilogy(qs_b, es_b, 'k.', ms=2, label='λ(n)', alpha=0.3)

    for k, results in all_results.items():
        qs = [r['q'] for r in results]
        es = [r['exp_sum_max_normed'] for r in results]
        ax.semilogy(qs, es, '.', color=colors[k], ms=2,
                    label=f'k={k}', alpha=0.4)

    # Reference lines
    ref1 = 1 / log(X_eff)**1.1
    ref2 = 1 / log(X_eff)**2
    ax.axhline(ref1, color='red', ls='--', alpha=0.5,
               label=f'1/(ln X)^{{1.1}} = {ref1:.4f}')
    ax.axhline(ref2, color='red', ls=':', alpha=0.5,
               label=f'1/(ln X)^2 = {ref2:.6f}')

    ax.set_xlabel('q', fontsize=13)
    ax.set_ylabel('max_{(a,q)=1} |ĝ(a/q)| / X', fontsize=13)
    ax.set_title('Minor-arc exponential sums (need to be below red lines for Goldbach)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plot2_exp_sums.png'), dpi=150)
    plt.close(fig)

    # --- Plot 3: Coprime vs non-coprime ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (k, results) in enumerate(all_results.items()):
        ax = axes[idx // 2][idx % 2]
        qs_c = [r['q'] for r in results if r.get('coprime_to_k', True)]
        qs_n = [r['q'] for r in results if not r.get('coprime_to_k', True)]
        Vs_c = [r['V_q'] for r in results if r.get('coprime_to_k', True)]
        Vs_n = [r['V_q'] for r in results if not r.get('coprime_to_k', True)]
        if qs_c:
            ax.semilogy(qs_c, Vs_c, '.', color='blue', ms=2,
                        label=f'(q,{k})=1', alpha=0.5)
        if qs_n:
            ax.semilogy(qs_n, Vs_n, '.', color='red', ms=3,
                        label=f'gcd(q,{k})>1', alpha=0.7)
        ax.set_title(f'k = {k}', fontsize=12)
        ax.set_xlabel('q')
        ax.set_ylabel('V(q)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Per-modulus variance: coprime vs non-coprime to k', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plot3_coprime_split.png'), dpi=150)
    plt.close(fig)

    # --- Plot 4: Per-q V(q) normalized ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for k, results in all_results.items():
        qs = np.array([r['q'] for r in results])
        Vn = np.array([r['V_q'] * r['q'] / X_eff**2 for r in results])
        ax.plot(qs, Vn, '.', color=colors[k], ms=2, label=f'k={k}', alpha=0.4)
    qs_b = np.array([r['q'] for r in baseline_results])
    Vn_b = np.array([r['V_q'] * r['q'] / X_eff**2 for r in baseline_results])
    ax.plot(qs_b, Vn_b, 'k.', ms=2, label='baseline λ', alpha=0.3)
    ax.set_xlabel('q', fontsize=13)
    ax.set_ylabel('V(q) · q / X²', fontsize=13)
    ax.set_title('Per-modulus normalized variance', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plot4_per_q_norm.png'), dpi=150)
    plt.close(fig)

    # --- Plot 5: Multi-scale (if available) ---
    if multi_scale_data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        q_test = 31  # a prime modulus
        # Left: |ĝ(a/q)|/X vs X for fixed q
        ax = axes[0]
        for key in ['baseline'] + SHIFTS:
            xs = sorted(multi_scale_data.get(key, {}).keys())
            vals = [multi_scale_data[key].get(x, {}).get(q_test, np.nan) for x in xs]
            c = colors.get(key, 'gray')
            lbl = 'λ (baseline)' if key == 'baseline' else f'k={key}'
            ax.loglog(xs, vals, 'o-', color=c, label=lbl, ms=5)

        # Reference: 1/sqrt(X/q) (random), 1/(ln X)^2 (Goldbach target)
        xs_ref = np.array(sorted(multi_scale_data.get('baseline', {}).keys()))
        ax.loglog(xs_ref, 1/np.sqrt(xs_ref/q_test), 'k--', alpha=0.3,
                  label=f'1/√(X/{q_test}) (random)')
        ax.loglog(xs_ref, 1/np.log(xs_ref)**2, 'r--', alpha=0.3,
                  label='1/(ln X)² (Goldbach)')

        ax.set_xlabel('X', fontsize=13)
        ax.set_ylabel(f'max |ĝ(a/{q_test})| / X', fontsize=13)
        ax.set_title(f'Scaling with X at q = {q_test}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Right: same for q = 97
        q_test2 = 97
        ax = axes[1]
        for key in ['baseline'] + SHIFTS:
            xs = sorted(multi_scale_data.get(key, {}).keys())
            vals = [multi_scale_data[key].get(x, {}).get(q_test2, np.nan) for x in xs]
            c = colors.get(key, 'gray')
            lbl = 'λ' if key == 'baseline' else f'k={key}'
            ax.loglog(xs, vals, 'o-', color=c, label=lbl, ms=5)

        ax.loglog(xs_ref, 1/np.sqrt(xs_ref/q_test2), 'k--', alpha=0.3,
                  label=f'1/√(X/{q_test2})')
        ax.loglog(xs_ref, 1/np.log(xs_ref)**2, 'r--', alpha=0.3,
                  label='1/(ln X)²')
        ax.set_xlabel('X', fontsize=13)
        ax.set_ylabel(f'max |ĝ(a/{q_test2})| / X', fontsize=13)
        ax.set_title(f'Scaling with X at q = {q_test2}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'plot5_multi_scale.png'), dpi=150)
        plt.close(fig)

    print("  Plots saved ✓")


# ============================================================
# MAIN
# ============================================================
def main():
    t_total = time.time()

    print("=" * 60)
    print(" KMT VARIANCE BOUND TEST FOR BINARY LIOUVILLE CORRELATIONS")
    print(f" X = {X:,}, Q_MAX = {Q_MAX}, shifts = {SHIFTS}")
    print(f" Output: {SAVE_DIR}")
    print("=" * 60)

    # --- Step 1: Sieve ---
    X_sieve = X + max(SHIFTS) + 1
    lam = compute_liouville(X_sieve)

    # --- Step 2: Multi-scale analysis (the key test) ---
    q_test_values = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                     53, 59, 67, 71, 79, 83, 89, 97, 101, 127, 151,
                     197, 211, 251, 307, 401, 499]
    if MULTI_SCALE:
        actual_scales = [s for s in SCALES if s <= X]
        ms_data = multi_scale_analysis(lam, actual_scales, SHIFTS, q_test_values)
    else:
        ms_data = None

    # --- Step 3: Full variance sweep at target X ---
    print(f"\n{'='*60}")
    print(f" FULL VARIANCE SWEEP AT X = {X:,}")
    print(f"{'='*60}")

    # Baseline
    g_base = lam[1:X+1].copy()
    baseline_results = variance_sweep(g_base, Q_MAX, "λ(n) baseline")
    del g_base

    # Binary correlations
    all_results = {}
    summary = {'X': X, 'Q_MAX': Q_MAX, 'shifts': SHIFTS, 'correlations': {}}

    for k in SHIFTS:
        print(f"\n{'='*60}")
        print(f" k = {k}: g(n) = λ(n)·λ(n+{k})")
        print(f"{'='*60}")

        g_k = (lam[1:X+1] * lam[1+k:X+1+k]).astype(np.int8)
        print(f"  Mean g_k = {g_k.mean():.8f}")

        results = variance_sweep(g_k, Q_MAX, f"g_k (k={k})", k=k)
        all_results[k] = results

        # Accumulate summary stats
        V_cop = sum(r['V_q'] for r in results if r.get('coprime_to_k', True))
        V_ncp = sum(r['V_q'] for r in results if not r.get('coprime_to_k', True))
        max_exp = max(r['exp_sum_max_normed'] for r in results)
        mean_exp = np.mean([r['exp_sum_max_normed'] for r in results])

        summary['correlations'][k] = {
            'R_final': float(results[-1]['R_Q']),
            'V_coprime': float(V_cop),
            'V_noncoprime': float(V_ncp),
            'max_exp_normed': float(max_exp),
            'mean_exp_normed': float(mean_exp),
        }

        del g_k

    # --- Step 4: Summary ---
    ref_11 = 1 / log(X)**1.1
    ref_2 = 1 / log(X)**2

    print(f"\n{'='*60}")
    print(f" RESULTS SUMMARY (X = {X:,})")
    print(f"{'='*60}")
    print(f"\n  Goldbach threshold: |ĝ(a/q)|/X < 1/(ln X)^{{1+ε}} ≈ {ref_11:.6f}")
    print(f"  Stronger threshold: |ĝ(a/q)|/X < 1/(ln X)^2    ≈ {ref_2:.6f}")

    print(f"\n  {'Function':<25s} {'R(Q_MAX)':<12s} {'max|ĝ|/X':<12s} {'mean|ĝ|/X':<12s}")
    print(f"  {'-'*61}")

    R_base = baseline_results[-1]['R_Q']
    max_e_b = max(r['exp_sum_max_normed'] for r in baseline_results)
    mean_e_b = np.mean([r['exp_sum_max_normed'] for r in baseline_results])
    print(f"  {'λ(n) (baseline)':<25s} {R_base:<12.6f} {max_e_b:<12.6f} {mean_e_b:<12.6f}")

    for k in SHIFTS:
        info = summary['correlations'][k]
        print(f"  {'λ(n)λ(n+'+str(k)+')':<25s} "
              f"{info['R_final']:<12.6f} "
              f"{info['max_exp_normed']:<12.6f} "
              f"{info['mean_exp_normed']:<12.6f}")

    print(f"\n  INTERPRETATION:")
    # Check if g_k exponential sums are comparable to baseline
    baseline_max = max_e_b
    all_below_threshold = True
    for k in SHIFTS:
        ratio = summary['correlations'][k]['max_exp_normed'] / baseline_max
        below = summary['correlations'][k]['max_exp_normed'] < ref_11
        if not below:
            all_below_threshold = False
        status = "✓ BELOW" if below else "✗ ABOVE"
        print(f"    k={k}: max|ĝ|/X ratio to baseline = {ratio:.2f}, "
              f"vs 1/(ln X)^{{1.1}}: {status}")

    if all_below_threshold:
        print(f"\n  ★ ALL correlations below Goldbach threshold at X={X:,}")
        print(f"    This supports input (c) of the circle method.")
    else:
        print(f"\n  ⚠ Some correlations above threshold — check multi-scale scaling")

    # --- Step 5: Save ---
    print(f"\n  Saving to {SAVE_DIR}...")
    with open(os.path.join(SAVE_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save multi-scale data
    if ms_data is not None:
        ms_serializable = {}
        for key, val in ms_data.items():
            ms_serializable[str(key)] = {str(x): v for x, v in val.items()}
        with open(os.path.join(SAVE_DIR, 'multi_scale.json'), 'w') as f:
            json.dump(ms_serializable, f, indent=2)

    # --- Step 6: Plots ---
    generate_plots(all_results, baseline_results, X, SAVE_DIR,
                   multi_scale_data=ms_data)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f" DONE. Total runtime: {elapsed/60:.1f} minutes")
    print(f" Results: {SAVE_DIR}")
    print(f"{'='*60}")

    return summary, all_results, baseline_results, ms_data


# ============================================================
if __name__ == '__main__':
    summary, all_results, baseline_results, ms_data = main()
