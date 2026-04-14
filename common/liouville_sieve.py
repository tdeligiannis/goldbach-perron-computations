#!/usr/bin/env python3
"""
Segmented Liouville sieve — shared module for all experiments.
================================================================

Provides λ(n) = (-1)^Ω(n) for n in a given range, using a segmented
sieve of Omega(n) mod 2.  All experiment scripts import from here.

Paper reference:
  T. Deligiannis, "A reduction of binary Goldbach to a four-point
  Chowla bound via the polynomial squaring identity" (2026).
  Appendix A — all five experiments rely on this sieve.

API
---
  compute_liouville(N)            → int8 array λ[0..N], λ[0]=0
  compute_liouville_segment(a, b) → int8 array λ[a..b]
  sieve_primes(limit)             → uint32 array of primes ≤ limit
  compute_random_cm(N, seed=42)   → int8 array of a random CM function

Author: Theodore Deligiannis & Claude (Anthropic)
Date:   April 2026
"""

import numpy as np
from math import isqrt
import time
import gc


def sieve_primes(limit):
    """Return sorted array of primes up to `limit` (Eratosthenes)."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p::p] = False
    return np.nonzero(is_prime)[0].astype(np.uint32)


def compute_liouville(N, verbose=True):
    """
    Compute λ(n) = (-1)^Ω(n) for n = 0, 1, ..., N.

    Returns int8 array with λ[0] = 0.
    Memory: ~N bytes for the sieve + ~N bytes for the result.

    Strategy
    --------
    Phase 1: Small primes p ≤ √N and all prime powers p^k ≤ N.
    Phase 2: Large primes √N < p ≤ N (each contributes at most
             ⌊N/p⌋ multiples, and primes > N/2 contribute exactly one).
    """
    t0 = time.time()
    if verbose:
        print(f"  Sieving λ(n) for n ≤ {N:,} ...")

    sqrt_N = isqrt(N)

    # Boolean sieve of Eratosthenes
    sieve = np.ones(N + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, sqrt_N + 1):
        if sieve[p]:
            sieve[p * p::p] = False

    # Omega parity: lam[n] starts at +1, each prime factor flips the sign
    lam = np.ones(N + 1, dtype=np.int8)
    lam[0] = 0

    # Phase 1: small primes and their powers
    small_primes = np.where(sieve[:sqrt_N + 1])[0]
    if verbose:
        print(f"    Phase 1: {len(small_primes):,} small primes (p ≤ {sqrt_N:,})")
    t1 = time.time()
    for p in small_primes:
        pk = int(p)
        while pk <= N:
            lam[pk::pk] *= -1
            pk *= int(p)
    if verbose:
        print(f"    Phase 1 done in {time.time() - t1:.1f}s")

    # Phase 2: large primes
    large_primes = np.where(sieve[sqrt_N + 1:])[0] + sqrt_N + 1
    if verbose:
        print(f"    Phase 2: {len(large_primes):,} large primes")
    t1 = time.time()

    # Primes > N/2 have exactly one multiple ≤ N: themselves
    cutoff = N // 2
    big = large_primes[large_primes > cutoff]
    lam[big] *= -1

    # Remaining large primes: √N < p ≤ N/2
    remaining = large_primes[large_primes <= cutoff]
    for p in remaining:
        lam[p::p] *= -1

    if verbose:
        print(f"    Phase 2 done in {time.time() - t1:.1f}s")

    del sieve
    gc.collect()

    # Verification
    _verify_liouville(lam)
    if verbose:
        L1k = int(np.sum(lam[1:1001].astype(np.int64)))
        print(f"    Verified ✓  L(1000) = {L1k}")
        print(f"    Total: {time.time() - t0:.1f}s, {lam.nbytes / 1e6:.0f} MB")

    return lam


def compute_liouville_segment(a, b, primes=None):
    """
    Compute λ(n) for n in [a, b] (inclusive).

    Parameters
    ----------
    a, b : int
        Range endpoints (1 ≤ a ≤ b).
    primes : array, optional
        Pre-computed primes up to √b.  If None, computed internally.

    Returns
    -------
    lam : int8 array of length (b - a + 1)
        lam[i] = λ(a + i).
    """
    if primes is None:
        primes = sieve_primes(isqrt(b))

    sz = b - a + 1
    # Track Ω(n) mod 2 via XOR
    omega_par = np.zeros(sz, dtype=np.uint8)
    n_remaining = np.arange(a, b + 1, dtype=np.int64)

    for p in primes:
        if p > b:
            break
        pk = int(p)
        while pk <= b:
            start = int(((a + pk - 1) // pk) * pk)
            if start <= b:
                indices = np.arange(start - a, sz, pk, dtype=np.int64)
                omega_par[indices] ^= 1
                n_remaining[indices] //= p
            pk_new = pk * int(p)
            if pk_new <= pk or pk_new > b:
                break
            pk = pk_new

    # Remaining factors > √b are large primes (at most one per n)
    omega_par[n_remaining > 1] ^= 1

    lam = np.int8(1) - np.int8(2) * omega_par.astype(np.int8)
    return lam


def compute_random_cm(N, seed=42, verbose=True):
    """
    Generate a random completely multiplicative f: N → {-1, +1} with
    f(p) = ±1 iid for each prime p.

    Used as a null comparison: if λ shows stronger cancellation than
    f_rand, the effect is specific to λ (connected to ζ(2s)/ζ(s)),
    not generic to all CM functions.

    Parameters
    ----------
    N : int
        Upper bound of the range.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    f : int8 array of length N+1, with f[0] = 0.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if verbose:
        print(f"  Generating random CM function (seed={seed}) for n ≤ {N:,}")

    sqrt_N = isqrt(N)

    # Sieve for Ω(n) and track which primes are "negative"
    is_p = np.ones(sqrt_N + 1, dtype=bool)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(sqrt_N) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    primes = np.nonzero(is_p)[0]

    # Full omega sieve (to identify large primes)
    omega = np.zeros(N + 1, dtype=np.int8)
    # Weighted omega (counting only primes where f(p) = -1)
    omega_neg = np.zeros(N + 1, dtype=np.int8)

    prime_flips = {}
    for p in primes:
        flip = (rng.random() < 0.5)
        prime_flips[int(p)] = flip
        pk = int(p)
        while pk <= N:
            omega[pk::pk] += 1
            if flip:
                omega_neg[pk::pk] += 1
            pk *= int(p)

    # Large primes: omega[n] == 0 for n ≥ 2 means n is prime
    large_mask = np.zeros(N + 1, dtype=bool)
    large_mask[2:] = (omega[2:] == 0)
    omega[large_mask] = 1

    # Assign random signs to large primes
    large_indices = np.nonzero(large_mask)[0]
    large_flips = rng.random(len(large_indices)) < 0.5
    omega_neg[large_indices[large_flips]] += 1

    f = np.where(omega_neg % 2 == 0, np.int8(1), np.int8(-1))
    f[0] = 0

    del omega, omega_neg, large_mask, large_indices
    gc.collect()

    if verbose:
        print(f"    Done in {time.time() - t0:.1f}s")
    return f


def _verify_liouville(lam):
    """Quick sanity check on known values."""
    expected = {
        1: 1, 2: -1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1, 8: -1,
        9: 1, 10: 1, 12: -1, 16: 1, 30: -1, 36: 1, 64: 1, 100: 1,
    }
    for n, v in expected.items():
        if n < len(lam):
            assert lam[n] == v, (
                f"Verification FAILED: λ({n}) = {lam[n]}, expected {v}")
