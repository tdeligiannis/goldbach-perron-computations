#!/usr/bin/env python3
"""
Experiment 5a: Bulk Cancellation — Polynomial Autocorrelation v3 (Bias-Corrected)
==================================================================================

Paper: T. Deligiannis, "A reduction of binary Goldbach to a
       four-point Chowla bound via the polynomial squaring identity"
       (2026), Appendix A, §A.5 (bulk cancellation).

Polynomial Autocorrelation of the Liouville Function — v3 (Bias-Corrected)
=============================================================================

CHANGES FROM v2:
  1. H_SHIFTS now uses ODD PRIMES (1,3,7,11,13,17,19,23,29,37) to avoid
     the local bias from shared small prime factors that contaminated even/
     composite shifts in v2.  The v2 run showed α≈3.6 for h=1 but α≈0
     for h=2,10,20 — the bias from gcd(n,n+h) sharing factors of h.
  
  2. BIAS CORRECTION: For each h, we estimate the asymptotic bias
     c_h = lim_{x→∞} (1/x)Σ λ(n)λ(n+h) from the tail of the data,
     then fit |Σ - c_h·x| / x instead of |Σ| / x.  This isolates the
     genuine cancellation rate from the constant floor.
  
  3. RANDOM CM COMPARISON: Generates a random completely multiplicative
     function f_rand with f(p) = ±1 (iid) for comparison.  If λ shows
     stronger cancellation than f_rand, the effect is specific to λ
     (connected to the zero of ζ(2s)/ζ(s) at s=1), not generic to CM
     functions.

KEY ALGORITHMIC INSIGHT (same as v2):
  By complete multiplicativity (Theorem 18.4):
      λ(n² + nh) = λ(n(n+h)) = λ(n) · λ(n+h)
  
  So we only need to sieve λ(m) for m up to X_MAX + h_max, NOT up to
  X_MAX · (X_MAX + h_max).

OBSERVABLES:
  1. BIAS-CORRECTED BULK CANCELLATION:
     |Σ_{n≤x} λ(n)λ(n+h) - c_h·x| / x  vs x
     → Fit exponent α in corrected |Σ|/x ~ (ln x)^{-α}
  
  2. UNCORRECTED BULK (as in v2, for comparison)
  
  3. NORMALIZED MEAN SQUARE: MS(x,H) × H  (1.0 = random baseline)
  
  4. RANDOM CM COMPARISON: same observables for f_rand

Hardware: Designed for 64-core, 540 GB RAM desktop.
Memory:   ~X_MAX bytes for sieve + ~8*X_MAX bytes per worker.

Usage:    Adjust X_MAX below, then run in Spyder.

Author: Theodore Deligiannis & Claude (Anthropic)
Date:   April 2026 — v3
"""

import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import time
import gc

# =============================================================================
# PARAMETERS
# =============================================================================

X_MAX       = 10_000_000_000   # 10^10 — needs ~10 GB sieve
# CHANGE 1: Odd primes only — avoids local bias from shared prime factors.
# v2 used [1,2,3,5,10,20,50,100,500,1000] and saw α≈0 for even/composite h.
# The bias comes from gcd(n,n+h) sharing factors of h: when p|h, the event
# p|n forces p|gcd(n,n+h), creating λ(p²)=+1 contributions that don't cancel.
# Odd primes avoid this (only one prime factor, and it's large enough that
# the 1/p bias is small).
H_SHIFTS    = [1, 3, 7, 11, 13, 17, 19, 23, 29, 37]
H_INTERVALS = [10, 30, 100, 300, 1000, 3000]  # short interval lengths
N_CORES     = min(64, cpu_count())  # parallel workers

# CHANGE 3: Run a random completely multiplicative comparison?
RUN_RANDOM_CM = True    # Set False to skip (saves ~50% runtime)
RANDOM_SEED   = 42      # for reproducibility

# Sampling: we don't store 10^9 MS values — sample at log-spaced points
N_BULK_SAMPLES = 5000   # points for bulk cancellation plot
N_MS_SAMPLES   = 2000   # points for MS plot


# =============================================================================
# 1. SIEVE: compute λ(n) for n = 0 .. N using additive Ω sieve
# =============================================================================

def compute_liouville(N):
    """
    Compute λ(n) = (-1)^Ω(n) for n = 0..N.
    Uses int8 omega array. Memory: N bytes.
    """
    omega = np.zeros(N + 1, dtype=np.int8)
    
    # Find primes up to sqrt(N)
    sieve_lim = int(sqrt(N)) + 1
    is_p = np.ones(sieve_lim, dtype=bool)
    is_p[0] = is_p[1] = False
    for i in range(2, int(sqrt(sieve_lim)) + 1):
        if is_p[i]:
            is_p[i*i::i] = False
    primes = np.nonzero(is_p)[0]
    
    n_primes = len(primes)
    t0 = time.time()
    t_last = t0
    
    for pi, p in enumerate(primes):
        pk = int(p)
        while pk <= N:
            omega[pk::pk] += 1
            pk *= int(p)
        
        t_now = time.time()
        if t_now - t_last > 5.0 or pi == n_primes - 1:
            pct = 100 * (pi + 1) / n_primes
            elapsed = t_now - t0
            eta = elapsed * (n_primes / (pi + 1) - 1) if pi > 0 else 0
            print(f"    [{pct:5.1f}%] p={p:,}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
            t_last = t_now
    
    # Fix large primes (omega == 0 for n >= 2 means prime)
    mask = np.zeros(N + 1, dtype=bool)
    mask[2:] = (omega[2:] == 0)
    omega[mask] = 1
    n_large = np.sum(mask)
    del mask
    
    # Convert: λ = (-1)^Ω  →  +1 if even, -1 if odd
    lam = np.where(omega % 2 == 0, np.int8(1), np.int8(-1))
    lam[0] = 0
    del omega
    gc.collect()
    
    print(f"    Fixed {n_large:,} large primes. Sieve complete.")
    return lam


# =============================================================================
# 1b. RANDOM COMPLETELY MULTIPLICATIVE FUNCTION (for comparison)
# =============================================================================

def compute_random_cm(N, seed=42):
    """
    Generate a random completely multiplicative function f with f(p) = ±1
    (independently for each prime). Uses the same sieve structure as λ.
    
    Purpose: If λ shows stronger polynomial decay than f_rand, the effect
    is specific to λ (connected to ζ(2s)/ζ(s)) rather than generic to all
    completely multiplicative functions.
    """
    rng = np.random.default_rng(seed)
    
    # Start with f(n) = +1 for all n
    f = np.ones(N + 1, dtype=np.int8)
    f[0] = 0
    
    # Sieve: for each prime p, assign random sign and propagate
    sieve_lim = int(sqrt(N)) + 1
    is_p = np.ones(sieve_lim, dtype=bool)
    is_p[0] = is_p[1] = False
    for i in range(2, int(sqrt(sieve_lim)) + 1):
        if is_p[i]:
            is_p[i*i::i] = False
    primes = np.nonzero(is_p)[0]
    
    # For each prime, assign f(p) = ±1 randomly, then f(p^k) = f(p)^k
    # We track the parity of the exponent of each prime in n's factorization
    # but with random signs instead of all -1.
    # 
    # Implementation: use omega-parity sieve, but flip sign only for primes
    # where f(p) = -1 (randomly chosen ~50% of primes).
    omega_weighted = np.zeros(N + 1, dtype=np.int8)
    
    for p in primes:
        sign = rng.choice([-1, 1])
        if sign == -1:
            # This prime contributes to the sign flip
            pk = int(p)
            while pk <= N:
                omega_weighted[pk::pk] += 1
                pk *= int(p)
    
    # Large primes (omega==0 after sieving small primes)
    # For large primes > sqrt(N), assign random sign
    large_prime_mask = np.zeros(N + 1, dtype=bool)
    large_prime_mask[2:] = True
    # Mark composites: any n with omega_weighted > 0 OR that was hit by the
    # full omega sieve.  Actually, we need to track ALL primes, not just
    # the randomly-chosen ones.  Let me use a cleaner approach.
    
    # Cleaner: sieve Ω(n) as before, then for each prime p, randomly decide
    # whether that prime's contribution flips the sign.
    del omega_weighted
    
    omega = np.zeros(N + 1, dtype=np.int8)
    prime_flips = {}  # p -> True if f(p) = -1
    
    for p in primes:
        prime_flips[int(p)] = (rng.random() < 0.5)  # 50% chance of -1
        pk = int(p)
        while pk <= N:
            omega[pk::pk] += 1
            pk *= int(p)
    
    # Fix large primes
    large_mask = np.zeros(N + 1, dtype=bool)
    large_mask[2:] = (omega[2:] == 0)
    omega[large_mask] = 1
    
    # For large primes, assign random signs
    large_indices = np.nonzero(large_mask)[0]
    
    # Now compute f(n) by tracking which prime factors have f(p) = -1.
    # f(n) = (-1)^{sum of exponents of primes p with f(p)=-1}
    # This requires a separate "weighted omega" sieve counting only
    # the primes where f(p) = -1.
    
    omega_neg = np.zeros(N + 1, dtype=np.int8)  # count exponents of "negative" primes
    
    for p in primes:
        if prime_flips[int(p)]:
            pk = int(p)
            while pk <= N:
                omega_neg[pk::pk] += 1
                pk *= int(p)
    
    # Large primes with f(p) = -1: randomly assign
    for lp in large_indices:
        if rng.random() < 0.5:
            omega_neg[lp] += 1
    
    f = np.where(omega_neg % 2 == 0, np.int8(1), np.int8(-1))
    f[0] = 0
    
    del omega, omega_neg, large_mask, large_indices
    gc.collect()
    
    return f


# =============================================================================
# 2. PER-SHIFT ANALYSIS (runs in parallel)
# =============================================================================

CHUNK_SIZE = 100_000_000  # 10^8 per chunk — fits in L3/RAM nicely

def analyze_shift(args):
    """
    For a single shift h, compute bulk cancellation and mean square.
    
    MEMORY-EFFICIENT: builds cumsum in chunks of 10^8 elements.
    Peak memory per worker: ~80 GB (the int64 cumsum array) for X_MAX=10^10.
    The λ array is shared (copy-on-write via fork).
    """
    h, x_max, H_list, n_bulk, n_ms = args
    lam = _global_lam
    
    results = {'h': h}
    
    # ── STEP A: Build global cumsum in chunks ──
    # cumsum[i] = Σ_{n=1}^{i+1} λ(n)·λ(n+h)
    # Memory: 8 bytes × x_max (e.g. 80 GB for 10^10).
    # The chunk loop avoids creating huge intermediate arrays.
    
    cumsum = np.empty(x_max, dtype=np.int64)
    running = np.int64(0)
    
    n_chunks = (x_max + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        lo = ci * CHUNK_SIZE          # 0-indexed position in cumsum
        hi = min(lo + CHUNK_SIZE, x_max)
        
        # λ(n)·λ(n+h) for n = lo+1 .. hi  (1-indexed)
        # int8 × int8 → int8 (no intermediate float64!)
        chunk_prod = lam[lo+1 : hi+1] * lam[lo+1+h : hi+1+h]
        
        # Cumsum within chunk, then add running offset
        np.cumsum(chunk_prod, dtype=np.int64, out=cumsum[lo:hi])
        cumsum[lo:hi] += running
        running = int(cumsum[hi - 1])
        
        del chunk_prod
    
    # ── OBSERVABLE 1: Bulk cancellation ──
    x_pts = np.unique(np.geomspace(100, x_max, n_bulk).astype(np.int64))
    x_pts = x_pts[(x_pts >= 100) & (x_pts <= x_max)]
    
    bulk_sum = cumsum[x_pts - 1].astype(np.float64)
    bulk_abs = np.abs(bulk_sum) / x_pts
    bulk_signed = bulk_sum / x_pts
    
    results['bulk_x'] = x_pts
    results['bulk_abs'] = bulk_abs
    results['bulk_signed'] = bulk_signed
    
    # ── CHANGE 2: BIAS CORRECTION ──
    # Estimate the asymptotic bias c_h from the tail of the signed data.
    # For h with shared small prime factors, the sum Σλ(n)λ(n+h)/x converges
    # to a nonzero constant c_h rather than 0.  We estimate c_h from the
    # last 30% of data points, then measure decay of |Σ - c_h·x| / x.
    #
    # For h = 1 (coprime shifts): c_h ≈ 0 (no bias).
    # For h = 2k (even shifts):   c_h ≈ product of local factors at p|h.
    
    tail_frac = 0.30  # use last 30% of data for bias estimation
    tail_start = int((1 - tail_frac) * len(x_pts))
    if tail_start < len(x_pts) - 10:
        c_h_estimate = np.median(bulk_signed[tail_start:])
    else:
        c_h_estimate = 0.0
    
    results['bias_estimate'] = c_h_estimate
    
    # Bias-corrected bulk cancellation
    bulk_corrected = np.abs(bulk_sum - c_h_estimate * x_pts.astype(np.float64)) / x_pts
    results['bulk_corrected'] = bulk_corrected
    
    # Fit exponent: |Σ|/x ~ a · (ln x)^{-α}
    ln_x = np.log(x_pts.astype(float))
    mask_fit = (bulk_abs > 0) & (ln_x > 5.0)
    if np.sum(mask_fit) > 50:
        try:
            ln_ln_x = np.log(ln_x[mask_fit])
            ln_bulk = np.log(bulk_abs[mask_fit])
            coeffs = np.polyfit(ln_ln_x, ln_bulk, 1)
            results['bulk_alpha'] = -coeffs[0]
            results['bulk_a'] = np.exp(coeffs[1])
            coeffs2 = np.polyfit(ln_ln_x, ln_bulk, 2)
            results['bulk_curvature'] = coeffs2[0]
        except Exception:
            results['bulk_alpha'] = None
    else:
        results['bulk_alpha'] = None
    
    # Fit exponent on BIAS-CORRECTED data
    mask_corr = (bulk_corrected > 0) & (ln_x > 5.0)
    if np.sum(mask_corr) > 50:
        try:
            ln_ln_x_c = np.log(ln_x[mask_corr])
            ln_bulk_c = np.log(bulk_corrected[mask_corr])
            coeffs_c = np.polyfit(ln_ln_x_c, ln_bulk_c, 1)
            results['corrected_alpha'] = -coeffs_c[0]
            coeffs2_c = np.polyfit(ln_ln_x_c, ln_bulk_c, 2)
            results['corrected_curvature'] = coeffs2_c[0]
        except Exception:
            results['corrected_alpha'] = None
    else:
        results['corrected_alpha'] = None
    
    # ── OBSERVABLE 2: Short-interval mean square (chunked) ──
    # We compute MS at sampled x-points without materializing a second
    # x_max-length array.  For each H, stream through cumsum in chunks
    # and accumulate Σ (window_sum/H)² in a running total.
    results['ms'] = {}
    
    # Prepend a zero for window computation: cs[0]=0, cs[1:]=cumsum
    # But we can't afford another 80 GB copy.  Instead, handle the
    # offset inline: window_sum(n) = cumsum[n+H-1] - cumsum[n-1]
    # where cumsum[-1] := 0  (i.e., for n=0, window_sum = cumsum[H-1]).
    
    for H in H_list:
        if H >= x_max // 2:
            continue
        
        n_windows = x_max - H + 1
        
        # Sample points for MS
        x_ms = np.unique(np.geomspace(max(200, 3*H), n_windows, n_ms).astype(int))
        x_ms = x_ms[(x_ms >= 1) & (x_ms < n_windows)]
        
        if len(x_ms) == 0:
            continue
        
        # For each sample point x, MS(x) = (1/x) Σ_{n=0}^{x-1} (win(n)/H)²
        # We compute this by streaming through chunks.
        running_sq_sum = 0.0
        ms_vals = np.empty(len(x_ms))
        sample_idx = 0   # pointer into x_ms
        
        for ci in range(0, n_windows, CHUNK_SIZE):
            ce = min(ci + CHUNK_SIZE, n_windows)
            
            # window_sum(n) = cumsum[n+H-1] - (cumsum[n-1] if n>0 else 0)
            right = cumsum[ci + H - 1 : ce + H - 1]  # view, no copy
            if ci == 0:
                left = np.empty(ce - ci, dtype=np.int64)
                left[0] = 0
                left[1:] = cumsum[0 : ce - 1]
            else:
                left = cumsum[ci - 1 : ce - 1]   # view
            
            chunk_win = (right - left).astype(np.float64)
            chunk_sq = (chunk_win / H) ** 2
            
            # Accumulate and sample
            chunk_cumsum_sq = np.cumsum(chunk_sq)
            chunk_cumsum_sq += running_sq_sum
            
            # Record MS at any sample points falling in this chunk
            while sample_idx < len(x_ms) and x_ms[sample_idx] <= ce:
                local_pos = x_ms[sample_idx] - ci - 1
                if 0 <= local_pos < len(chunk_cumsum_sq):
                    ms_vals[sample_idx] = chunk_cumsum_sq[local_pos] / x_ms[sample_idx]
                sample_idx += 1
            
            running_sq_sum = float(chunk_cumsum_sq[-1])
            del chunk_win, chunk_sq, chunk_cumsum_sq, left
        
        ms_normalized = ms_vals[:sample_idx] * H
        
        results['ms'][H] = {
            'x': x_ms[:sample_idx],
            'ms': ms_vals[:sample_idx],
            'ms_norm': ms_normalized,
        }
    
    # ── OBSERVABLE 3: Autocorrelation at small lags ──
    n_auto = min(x_max, 10_000_000)
    v = (lam[1:n_auto+1].astype(np.float64) *
         lam[1+h:n_auto+1+h].astype(np.float64))
    
    max_lag = 200
    autocorr = np.zeros(max_lag)
    for d in range(max_lag):
        autocorr[d] = np.mean(v[:n_auto-d] * v[d:n_auto]) if d > 0 else np.mean(v*v)
    
    results['autocorr'] = autocorr
    results['autocorr_n'] = n_auto
    del v
    
    # Free the big array
    del cumsum
    gc.collect()
    
    return results


# Global reference to λ array for multiprocessing (shared via fork)
_global_lam = None

def init_worker(lam_array):
    global _global_lam
    _global_lam = lam_array


# =============================================================================
# 3. MODEL FITTING
# =============================================================================

def fit_bulk_models(x_pts, bulk_abs):
    """Fit polynomial and subexponential models to bulk cancellation."""
    ln_x = np.log(x_pts.astype(float))
    mask = (bulk_abs > 0) & (ln_x > 4.0)
    ln_x_f = ln_x[mask]
    b_f = bulk_abs[mask]
    
    if len(b_f) < 50:
        return {}
    
    results = {'ln_x': ln_x_f, 'data': b_f}
    
    # Model A: |Σ|/x ~ a · (ln x)^{-α}
    try:
        def mA(lx, alpha, a):
            return a * lx ** (-alpha)
        popt, _ = curve_fit(mA, ln_x_f, b_f, p0=[1.5, 1.0],
                            bounds=([0.01, 1e-10], [10., 1e5]), maxfev=20000)
        fitted = mA(ln_x_f, *popt)
        ss_res = np.sum((b_f - fitted)**2)
        ss_tot = np.sum((b_f - np.mean(b_f))**2)
        results['A'] = {'alpha': popt[0], 'a': popt[1],
                        'R2': 1 - ss_res/ss_tot if ss_tot > 0 else 0,
                        'fitted': fitted}
    except Exception:
        pass
    
    # Model B: |Σ|/x ~ a · exp(-c · (ln ln x)^{1/3})
    try:
        def mB(lx, c, a):
            return a * np.exp(-c * np.log(np.maximum(lx, 1.01))**(1/3))
        popt, _ = curve_fit(mB, ln_x_f, b_f, p0=[1.0, 1.0],
                            bounds=([0.01, 1e-10], [50., 1e5]), maxfev=20000)
        fitted = mB(ln_x_f, *popt)
        ss_res = np.sum((b_f - fitted)**2)
        ss_tot = np.sum((b_f - np.mean(b_f))**2)
        results['B'] = {'c': popt[0], 'a': popt[1],
                        'R2': 1 - ss_res/ss_tot if ss_tot > 0 else 0,
                        'fitted': fitted}
    except Exception:
        pass
    
    return results


def fit_ms_decay(x_pts, ms_norm):
    """Fit decay of normalized MS (MS×H) from 1.0."""
    ln_x = np.log(x_pts.astype(float))
    mask = (ms_norm > 0) & (ms_norm < 2.0) & (ln_x > 4.0) & np.isfinite(ms_norm)
    ln_x_f = ln_x[mask]
    mn_f = ms_norm[mask]
    
    if len(mn_f) < 50:
        return {}
    
    results = {'ln_x': ln_x_f, 'data': mn_f}
    
    # Model: MS×H ~ a · (ln x)^{-c} + b  (decaying from baseline ≈ 1)
    try:
        def decay_model(lx, c, a, b):
            return a * lx ** (-c) + b
        popt, _ = curve_fit(decay_model, ln_x_f, mn_f,
                            p0=[0.5, 1.0, 0.9],
                            bounds=([0.0, 0.0, 0.0], [10., 100., 2.0]),
                            maxfev=20000)
        fitted = decay_model(ln_x_f, *popt)
        ss_res = np.sum((mn_f - fitted)**2)
        ss_tot = np.sum((mn_f - np.mean(mn_f))**2)
        results['fit'] = {'c': popt[0], 'a': popt[1], 'b': popt[2],
                          'R2': 1 - ss_res/ss_tot if ss_tot > 0 else 0,
                          'fitted': fitted}
    except Exception:
        pass
    
    return results


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    t_global = time.time()
    global _global_lam
    
    sieve_max = X_MAX + max(H_SHIFTS)
    
    print("=" * 74)
    print("  POLYNOMIAL AUTOCORRELATION v3 — BIAS-CORRECTED + ODD PRIMES")
    print("  λ(n²+nh) = λ(n)·λ(n+h)  via complete multiplicativity")
    print("=" * 74)
    print(f"\n  Configuration:")
    print(f"    X_MAX         = {X_MAX:,}")
    print(f"    Sieve range   = {sieve_max:,}  ({sieve_max/1e9:.2f} GB)")
    print(f"    h shifts      = {H_SHIFTS}")
    print(f"    H intervals   = {H_INTERVALS}")
    print(f"    Cores         = {N_CORES}")
    print(f"    ln(X_MAX)     = {log(X_MAX):.3f}")
    print(f"    ln(ln(X_MAX)) = {log(log(X_MAX)):.3f}")
    print(f"    (ln X)^{{-4}}   = {log(X_MAX)**(-4):.2e}  (MR target)")
    print(f"    1/H_min       = {1/min(H_INTERVALS):.4f}  (CLT baseline)")
    
    # ------------------------------------------------------------------
    # STEP 1: SIEVE
    # ------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  STEP 1: Sieve λ(n) for n = 0..{sieve_max:,}")
    print(f"{'='*74}")
    t0 = time.time()
    lam = compute_liouville(sieve_max)
    t_sieve = time.time() - t0
    print(f"  Sieve time: {t_sieve:.1f}s ({t_sieve/60:.1f} min)")
    
    # Sanity check
    expected = {1:1, 2:-1, 3:-1, 4:1, 5:-1, 6:1, 7:-1, 8:-1, 9:1, 10:1}
    for n, v in expected.items():
        assert lam[n] == v, f"λ({n})={lam[n]}, expected {v}"
    print(f"  Sanity check: passed")
    
    _global_lam = lam  # set global for workers
    
    # ------------------------------------------------------------------
    # STEP 2: PARALLEL COMPUTATION
    # ------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  STEP 2: Computing observables ({len(H_SHIFTS)} shifts × "
          f"{len(H_INTERVALS)} intervals) on {N_CORES} cores")
    print(f"{'='*74}")
    
    # Compute safe parallelism: each worker needs ~8*X_MAX bytes for cumsum
    per_worker_gb = 8 * X_MAX / 1e9
    available_gb = 500  # conservative estimate for 540 GB system
    sieve_gb = sieve_max / 1e9
    max_parallel = max(1, int((available_gb - sieve_gb) / max(per_worker_gb, 0.1)))
    max_parallel = min(max_parallel, len(H_SHIFTS), N_CORES)
    
    print(f"    Per-worker memory   ≈ {per_worker_gb:.1f} GB (cumsum array)")
    print(f"    Max parallel workers = {max_parallel}")
    
    tasks = [(h, X_MAX, H_INTERVALS, N_BULK_SAMPLES, N_MS_SAMPLES)
             for h in H_SHIFTS]
    
    t0 = time.time()
    
    # Use fork-based pool; λ array is shared copy-on-write
    with Pool(processes=max_parallel,
              initializer=init_worker, initargs=(lam,)) as pool:
        all_results = []
        for i, result in enumerate(pool.imap_unordered(analyze_shift, tasks)):
            h = result['h']
            alpha = result.get('bulk_alpha')
            alpha_str = f"α={alpha:.3f}" if alpha else "α=N/A"
            print(f"    [{i+1}/{len(tasks)}] h={h:5d} done  ({alpha_str})",
                  flush=True)
            all_results.append(result)
    
    # Sort by h
    all_results.sort(key=lambda r: r['h'])
    
    t_compute = time.time() - t0
    print(f"\n  Computation time: {t_compute:.1f}s ({t_compute/60:.1f} min)")
    
    # Free sieve
    del lam, _global_lam
    gc.collect()
    
    # ------------------------------------------------------------------
    # STEP 2b: RANDOM CM COMPARISON (optional)
    # ------------------------------------------------------------------
    rand_results = []
    t_rand = 0
    
    if RUN_RANDOM_CM:
        print(f"\n{'='*74}")
        print(f"  STEP 2b: Random CM comparison (seed={RANDOM_SEED})")
        print(f"{'='*74}")
        
        t0r = time.time()
        print(f"    Generating random CM function f_rand(p) = ±1 iid...")
        f_rand = compute_random_cm(sieve_max, seed=RANDOM_SEED)
        print(f"    Random CM sieve done in {time.time()-t0r:.1f}s")
        
        # Sanity: f_rand should be ±1 everywhere (except 0)
        assert np.all(np.abs(f_rand[1:100]) == 1), "Random CM sanity failed"
        
        _global_lam = f_rand  # reuse the same worker infrastructure
        
        # Run on a subset of h values from H_SHIFTS (must be within sieve range)
        h_rand = [h for h in H_SHIFTS if h <= 13][:3]  # first 3 small shifts
        if not h_rand:
            h_rand = H_SHIFTS[:3]
        tasks_rand = [(h, X_MAX, [100, 1000], N_BULK_SAMPLES, N_MS_SAMPLES)
                      for h in h_rand]
        
        max_par_rand = max(1, min(len(h_rand),
                          int((available_gb - sieve_gb) / max(per_worker_gb, 0.1)),
                          N_CORES))
        
        with Pool(processes=max_par_rand,
                  initializer=init_worker, initargs=(f_rand,)) as pool:
            for i, result in enumerate(pool.imap_unordered(analyze_shift, tasks_rand)):
                h = result['h']
                alpha = result.get('bulk_alpha')
                alpha_str = f"α={alpha:.3f}" if alpha else "α=N/A"
                print(f"    [rand {i+1}/{len(h_rand)}] h={h:5d} done  ({alpha_str})")
                rand_results.append(result)
        
        rand_results.sort(key=lambda r: r['h'])
        t_rand = time.time() - t0r
        print(f"    Random CM time: {t_rand:.1f}s ({t_rand/60:.1f} min)")
        
        del f_rand, _global_lam
        gc.collect()
    else:
        print(f"\n  (Random CM comparison skipped — set RUN_RANDOM_CM=True to enable)")
    
    # ------------------------------------------------------------------
    # STEP 3: ANALYSIS
    # ------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  STEP 3: Model fitting and analysis")
    print(f"{'='*74}")
    
    print(f"\n  ── Bulk cancellation: |Σ λ(n)λ(n+h)| / x ──")
    print(f"  {'h':>5s}  {'α raw':>8s} {'R²_A':>7s}  "
          f"{'α corr':>8s}  {'bias':>9s}  "
          f"{'c (exp)':>9s} {'R²_B':>7s}  {'Winner':>8s}")
    print(f"  {'─'*75}")
    
    bulk_fits = {}
    for r in all_results:
        h = r['h']
        fit = fit_bulk_models(r['bulk_x'], r['bulk_abs'])
        bulk_fits[h] = fit
        
        # Also fit bias-corrected data
        fit_corr = fit_bulk_models(r['bulk_x'], r['bulk_corrected'])
        bulk_fits[f'{h}_corr'] = fit_corr
        
        alpha_s = f"{fit['A']['alpha']:.4f}" if fit.get('A') else '—'
        r2a_s = f"{fit['A']['R2']:.4f}" if fit.get('A') else '—'
        c_s = f"{fit['B']['c']:.4f}" if fit.get('B') else '—'
        r2b_s = f"{fit['B']['R2']:.4f}" if fit.get('B') else '—'
        
        alpha_corr = r.get('corrected_alpha')
        alpha_corr_s = f"{alpha_corr:.4f}" if alpha_corr else '—'
        bias_s = f"{r.get('bias_estimate', 0):.6f}"
        
        r2a = fit['A']['R2'] if fit.get('A') else -999
        r2b = fit['B']['R2'] if fit.get('B') else -999
        winner = 'Poly' if r2a > r2b else 'Exp'
        
        print(f"  {h:5d}  {alpha_s:>8s} {r2a_s:>7s}  "
              f"{alpha_corr_s:>8s}  {bias_s:>9s}  "
              f"{c_s:>9s} {r2b_s:>7s}  {winner:>8s}")
    
    # Aggregate
    alphas_raw = [bulk_fits[h]['A']['alpha'] for h in H_SHIFTS
                  if bulk_fits.get(h, {}).get('A')]
    alphas_corr = [r.get('corrected_alpha') for r in all_results
                   if r.get('corrected_alpha') is not None]
    
    if alphas_raw:
        print(f"\n  Raw α statistics:       mean={np.mean(alphas_raw):.3f}, "
              f"median={np.median(alphas_raw):.3f}, std={np.std(alphas_raw):.3f}")
    if alphas_corr:
        print(f"  Corrected α statistics: mean={np.mean(alphas_corr):.3f}, "
              f"median={np.median(alphas_corr):.3f}, std={np.std(alphas_corr):.3f}")
    print(f"  (MR target for Goldbach: α ≈ 2+c for the bulk sum)")
    
    # Random CM comparison
    if rand_results:
        print(f"\n  ── Random CM comparison (f_rand, seed={RANDOM_SEED}) ──")
        for r in rand_results:
            h = r['h']
            alpha = r.get('bulk_alpha')
            alpha_corr = r.get('corrected_alpha')
            bias = r.get('bias_estimate', 0)
            a_str = f"{alpha:.3f}" if alpha else "N/A"
            ac_str = f"{alpha_corr:.3f}" if alpha_corr else "N/A"
            print(f"    h={h:5d}:  α_raw={a_str:>7s}  "
                  f"α_corr={ac_str:>7s}  bias={bias:.6f}")
        print(f"  If λ has larger α than f_rand → cancellation specific to λ")
    
    # MS analysis
    print(f"\n  ── Normalized mean square: MS × H ──")
    print(f"  (Should be ~1.0 for random; decay below 1.0 = cancellation)")
    
    for r in all_results[:4]:  # first few h values
        h = r['h']
        for H in H_INTERVALS[:3]:
            if H in r['ms']:
                mn = r['ms'][H]['ms_norm']
                if len(mn) > 0:
                    final_val = np.mean(mn[-100:]) if len(mn) > 100 else mn[-1]
                    print(f"    h={h:4d}, H={H:5d}: MS×H final ≈ {final_val:.6f}")
    
    # ------------------------------------------------------------------
    # STEP 4: PLOTS
    # ------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"  STEP 4: Generating plots")
    print(f"{'='*74}")
    
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})
    cmap = plt.cm.tab10
    
    # === FIGURE 1: BULK CANCELLATION (the key observable) ===
    n_h = len(H_SHIFTS)
    ncols = min(5, n_h)
    nrows = (n_h + ncols - 1) // ncols
    
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                                squeeze=False)
    fig1.suptitle(
        r'Bulk cancellation: $|\sum_{n\leq x} \lambda(n)\lambda(n{+}h)|\,/\,x$'
        '\nwith polynomial and subexponential fits',
        fontsize=14, y=1.01)
    
    for idx, r in enumerate(all_results):
        ax = axes1[idx // ncols, idx % ncols]
        h = r['h']
        
        # Data
        step = max(1, len(r['bulk_x']) // 500)
        x_p = r['bulk_x'][::step]
        b_p = r['bulk_abs'][::step]
        ax.loglog(x_p, b_p, '.', ms=1, alpha=0.3, color='gray')
        
        # Fits
        fit = bulk_fits.get(h, {})
        if fit.get('A'):
            ax.loglog(np.exp(fit['ln_x']), fit['A']['fitted'], 'b-', lw=2,
                      label=r'$(\ln x)^{-%.2f}$  $R^2$=%.3f'
                      % (fit['A']['alpha'], fit['A']['R2']))
        if fit.get('B'):
            ax.loglog(np.exp(fit['ln_x']), fit['B']['fitted'], 'r--', lw=2,
                      label=r'$e^{-%.1f u^{1/3}}$  $R^2$=%.3f'
                      % (fit['B']['c'], fit['B']['R2']))
        
        # Reference lines
        xr = np.logspace(2, np.log10(X_MAX), 200)
        lxr = np.log(xr)
        ax.loglog(xr, 1/lxr**2, 'b:', alpha=0.3, lw=1)
        ax.loglog(xr, 1/lxr**4, 'purple', alpha=0.2, lw=1, ls=':')
        
        ax.set_title(f'h = {h}')
        ax.set_xlabel('x')
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.15, which='both')
    
    for idx in range(n_h, nrows * ncols):
        axes1[idx // ncols, idx % ncols].set_visible(False)
    fig1.tight_layout()
    fig1.savefig('fig1_bulk_cancellation.png', dpi=150, bbox_inches='tight')
    print("  Saved fig1_bulk_cancellation.png")
    
    # === FIGURE 2: LOG-LOG DIAGNOSTIC (ln(|Σ|/x) vs ln ln x) ===
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                                squeeze=False)
    fig2.suptitle(
        r'Key diagnostic: $\ln(|\Sigma|/x)$ vs $\ln\ln x$'
        '\nLinear = polynomial | Curvature = subexponential',
        fontsize=14, y=1.01)
    
    for idx, r in enumerate(all_results):
        ax = axes2[idx // ncols, idx % ncols]
        h = r['h']
        
        mask = (r['bulk_abs'] > 0) & (r['bulk_x'] > np.e**4)
        if np.sum(mask) > 50:
            lnlnx = np.log(np.log(r['bulk_x'][mask].astype(float)))
            lnb = np.log(r['bulk_abs'][mask])
            
            step = max(1, len(lnlnx) // 500)
            ax.scatter(lnlnx[::step], lnb[::step], s=1, alpha=0.2, color='gray')
            
            # Linear, quadratic, cubic fits
            c1 = np.polyfit(lnlnx, lnb, 1)
            c2 = np.polyfit(lnlnx, lnb, 2)
            
            lnlnx_s = np.linspace(lnlnx.min(), lnlnx.max(), 200)
            ax.plot(lnlnx_s, np.polyval(c1, lnlnx_s), 'b-', lw=2.5,
                    label=f'slope = {c1[0]:.3f}')
            ax.plot(lnlnx_s, np.polyval(c2, lnlnx_s), 'r--', lw=1.5,
                    label=f'curvature = {c2[0]:.3f}')
        
        ax.set_xlabel(r'$\ln\ln x$')
        ax.set_ylabel(r'$\ln(|\Sigma|/x)$')
        ax.set_title(f'h = {h}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    for idx in range(n_h, nrows * ncols):
        axes2[idx // ncols, idx % ncols].set_visible(False)
    fig2.tight_layout()
    fig2.savefig('fig2_loglog_diagnostic.png', dpi=150, bbox_inches='tight')
    print("  Saved fig2_loglog_diagnostic.png")
    
    # === FIGURE 3: NORMALIZED MEAN SQUARE (MS × H) ===
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                                squeeze=False)
    fig3.suptitle(
        r'Normalized mean square: $\mathrm{MS}(x,H) \times H$'
        '\n= 1.0 for random ±1;  decay below 1.0 = polynomial cancellation',
        fontsize=14, y=1.01)
    
    for idx, r in enumerate(all_results):
        ax = axes3[idx // ncols, idx % ncols]
        h = r['h']
        
        for Hi, H in enumerate(H_INTERVALS):
            if H in r['ms']:
                d = r['ms'][H]
                step = max(1, len(d['x']) // 300)
                ax.semilogx(d['x'][::step], d['ms_norm'][::step],
                            '-', lw=0.8, alpha=0.7, color=cmap(Hi),
                            label=f'H={H}')
        
        ax.axhline(1.0, color='k', lw=1, ls='--', alpha=0.5, label='Random baseline')
        ax.set_xlabel('x')
        ax.set_ylabel('MS × H')
        ax.set_title(f'h = {h}')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0.0, max(1.5, ax.get_ylim()[1]))
    
    for idx in range(n_h, nrows * ncols):
        axes3[idx // ncols, idx % ncols].set_visible(False)
    fig3.tight_layout()
    fig3.savefig('fig3_normalized_ms.png', dpi=150, bbox_inches='tight')
    print("  Saved fig3_normalized_ms.png")
    
    # === FIGURE 4: BULK CANCELLATION OVERLAY (all h on one plot) ===
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    ax4.set_title(
        r'Bulk cancellation: $|\sum_{n\leq x}\lambda(n)\lambda(n{+}h)|\,/\,x$'
        '  (all shifts overlaid)',
        fontsize=14)
    
    for idx, r in enumerate(all_results):
        h = r['h']
        step = max(1, len(r['bulk_x']) // 800)
        ax4.loglog(r['bulk_x'][::step], r['bulk_abs'][::step],
                   '.', ms=1.5, alpha=0.35, color=cmap(idx % 10),
                   label=f'h={h}')
    
    xr = np.logspace(2, np.log10(X_MAX), 500)
    lxr = np.log(xr)
    llxr = np.log(np.maximum(lxr, 1.01))
    
    ax4.loglog(xr, 1/lxr, 'b-', lw=3, alpha=0.4, label=r'$(\ln x)^{-1}$')
    ax4.loglog(xr, 1/lxr**2, 'b--', lw=2.5, alpha=0.4, label=r'$(\ln x)^{-2}$')
    ax4.loglog(xr, 1/lxr**4, color='purple', lw=2, alpha=0.3, ls=':',
               label=r'$(\ln x)^{-4}$ (MR)')
    ax4.loglog(xr, np.exp(-0.5*llxr**(1/3)), 'r-', lw=2.5, alpha=0.4,
               label=r'$e^{-0.5(\ln\ln x)^{1/3}}$')
    ax4.loglog(xr, 0.5/np.sqrt(xr), 'k:', lw=1, alpha=0.3,
               label=r'$x^{-1/2}$ (random walk)')
    
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel(r'$|\sum \lambda(n)\lambda(n{+}h)| / x$', fontsize=12)
    ax4.legend(fontsize=8, ncol=2, loc='upper right')
    ax4.grid(True, alpha=0.15, which='both')
    fig4.tight_layout()
    fig4.savefig('fig4_bulk_overlay.png', dpi=150, bbox_inches='tight')
    print("  Saved fig4_bulk_overlay.png")
    
    # === FIGURE 5: EXPONENT α vs h ===
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig5.suptitle('Summary: fitted exponents and model comparison', fontsize=14)
    
    h_vals = [r['h'] for r in all_results]
    alpha_vals = [bulk_fits[h]['A']['alpha'] if bulk_fits.get(h,{}).get('A') else np.nan
                  for h in h_vals]
    r2a_vals = [bulk_fits[h]['A']['R2'] if bulk_fits.get(h,{}).get('A') else np.nan
                for h in h_vals]
    r2b_vals = [bulk_fits[h]['B']['R2'] if bulk_fits.get(h,{}).get('B') else np.nan
                for h in h_vals]
    
    # Panel a: α vs h
    ax5a.plot(h_vals, alpha_vals, 'bo-', ms=8, lw=2)
    ax5a.axhline(2, color='purple', ls='--', lw=1.5, alpha=0.5, label='α=2')
    ax5a.axhline(4, color='red', ls='--', lw=1.5, alpha=0.5, label='α=4 (MR)')
    ax5a.set_xlabel('Shift h')
    ax5a.set_ylabel(r'Exponent $\alpha$')
    ax5a.set_title(r'Bulk: $|\Sigma|/x \sim (\ln x)^{-\alpha}$')
    ax5a.set_xscale('log')
    ax5a.legend(fontsize=9)
    ax5a.grid(True, alpha=0.2)
    
    # Panel b: R² comparison
    ax5b.plot(h_vals, r2a_vals, 'bo-', label='Model A (poly)', ms=7)
    ax5b.plot(h_vals, r2b_vals, 'rs--', label='Model B (exp)', ms=7)
    ax5b.set_xlabel('Shift h')
    ax5b.set_ylabel(r'$R^2$')
    ax5b.set_title('Goodness of fit')
    ax5b.set_xscale('log')
    ax5b.legend(fontsize=9)
    ax5b.grid(True, alpha=0.2)
    
    # Panel c: Autocorrelation decay
    for idx, r in enumerate(all_results[:5]):
        ac = r['autocorr']
        ac_norm = ac / ac[0]  # normalize C(0) = 1
        ax5c.plot(range(len(ac)), ac_norm, '-', alpha=0.7,
                  color=cmap(idx), label=f'h={r["h"]}')
    ax5c.axhline(0, color='k', lw=0.5)
    ax5c.set_xlabel('Lag d')
    ax5c.set_ylabel('Normalized autocorrelation')
    ax5c.set_title(r'$C(d)/C(0)$ for $\lambda(n)\lambda(n{+}h)$')
    ax5c.set_xlim(0, 100)
    ax5c.legend(fontsize=8)
    ax5c.grid(True, alpha=0.2)
    
    fig5.tight_layout()
    fig5.savefig('fig5_summary.png', dpi=150, bbox_inches='tight')
    print("  Saved fig5_summary.png")
    
    # === FIGURE 6: CONVERGENCE — α as function of x_cutoff ===
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))
    fig6.suptitle('Convergence of fitted exponent α with increasing x',
                  fontsize=14)
    
    for idx, r in enumerate(all_results[:6]):
        h = r['h']
        mask = (r['bulk_abs'] > 0) & (r['bulk_x'] > np.e**4)
        if np.sum(mask) < 100:
            continue
        
        x_f = r['bulk_x'][mask]
        b_f = r['bulk_abs'][mask]
        lnlnx_f = np.log(np.log(x_f.astype(float)))
        lnb_f = np.log(b_f)
        
        # Fit using data up to various cutoffs
        cutoffs = np.linspace(len(x_f) // 5, len(x_f), 20, dtype=int)
        alpha_run = []
        x_cut_vals = []
        
        for cut in cutoffs:
            if cut < 50:
                continue
            c1 = np.polyfit(lnlnx_f[:cut], lnb_f[:cut], 1)
            alpha_run.append(-c1[0])
            x_cut_vals.append(x_f[cut - 1])
        
        if alpha_run:
            ax6a.plot(np.log(np.array(x_cut_vals, dtype=float)),
                      alpha_run, 'o-', ms=4, lw=1.5,
                      color=cmap(idx), label=f'h={h}')
    
    ax6a.axhline(2, color='purple', ls='--', lw=1.5, alpha=0.5)
    ax6a.axhline(4, color='red', ls='--', lw=1.5, alpha=0.5)
    ax6a.set_xlabel(r'$\ln(x_{\mathrm{cutoff}})$')
    ax6a.set_ylabel(r'Fitted $\alpha$')
    ax6a.set_title(r'$\alpha$ convergence (bulk sum)')
    ax6a.legend(fontsize=8)
    ax6a.grid(True, alpha=0.2)
    
    # Panel b: signed sum — does it oscillate or drift?
    for idx, r in enumerate(all_results[:6]):
        h = r['h']
        step = max(1, len(r['bulk_x']) // 500)
        ax6b.semilogx(r['bulk_x'][::step],
                       r['bulk_signed'][::step],
                       '-', lw=0.8, alpha=0.6, color=cmap(idx),
                       label=f'h={h}')
    
    ax6b.axhline(0, color='k', lw=0.5)
    ax6b.set_xlabel('x')
    ax6b.set_ylabel(r'$\sum \lambda(n)\lambda(n{+}h) / x$  (signed)')
    ax6b.set_title('Signed bulk average (bias detection)')
    ax6b.legend(fontsize=8)
    ax6b.grid(True, alpha=0.2)
    
    fig6.tight_layout()
    fig6.savefig('fig6_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved fig6_convergence.png")
    
    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    t_total = time.time() - t_global
    
    print(f"\n{'='*74}")
    print(f"  SUMMARY")
    print(f"{'='*74}")
    print(f"""
  Total runtime: {t_total:.0f}s ({t_total/60:.1f} min)
  Sieve: {t_sieve:.0f}s  |  Computation: {t_compute:.0f}s  |  Random CM: {t_rand:.0f}s
  
  Range: x up to {X_MAX:,}
    ln(X_MAX)     = {log(X_MAX):.3f}
    ln(ln(X_MAX)) = {log(log(X_MAX)):.3f}
    (ln X)^{{-4}}   = {log(X_MAX)**(-4):.2e}

  KEY RESULT — Bulk cancellation exponents α:
  (RAW = uncorrected, CORR = bias-subtracted)
""")
    for r in all_results:
        h = r['h']
        alpha = r.get('bulk_alpha')
        alpha_c = r.get('corrected_alpha')
        bias = r.get('bias_estimate', 0)
        if alpha is not None:
            corr_str = f"α_corr={alpha_c:.2f}" if alpha_c else "α_corr=N/A"
            print(f"    h = {h:5d}:  α_raw = {alpha:.4f}  {corr_str}  "
                  f"bias = {bias:.6f}")
    
    if rand_results:
        print(f"\n  RANDOM CM COMPARISON (f_rand, seed={RANDOM_SEED}):")
        for r in rand_results:
            h = r['h']
            alpha = r.get('bulk_alpha')
            a_str = f"{alpha:.4f}" if alpha else "N/A"
            print(f"    h = {h:5d}:  α_rand = {a_str}")
    
    print(f"""
  INTERPRETATION:
    • CHANGE 1 (odd primes): If α is now consistently positive across
      all h (unlike v2 where even h gave α≈0), the v2 flatness was
      caused by local bias, not by absence of cancellation.
    
    • CHANGE 2 (bias correction): If α_corr >> α_raw for composite h,
      the bias was masking the true decay rate.
    
    • CHANGE 3 (random CM): If α_λ >> α_rand, the cancellation is
      specific to λ (ζ(2s)/ζ(s) zero at s=1), not generic to CM functions.
      If α_λ ≈ α_rand, all CM functions with |f(p)|=1 have this property.

  Figure 6 (convergence) shows whether α stabilizes as x grows.
""")
    
    plt.show()
    print("Done. All 6 figures saved.")


if __name__ == '__main__':
    main()
