#!/usr/bin/env python3
"""
Experiment 4: Wavelet Verification of Pointwise Cancellation
==============================================================

Paper: T. Deligiannis, "A reduction of binary Goldbach to a
       four-point Chowla bound via the polynomial squaring identity"
       (2026), Appendix A, §A.4, and Companion Paper §4.

Computes wavelet coefficients C_j(k, h) for the binary correlation
g(n) = λ(n)·λ(n+h) at dyadic scales j, where

  C_j(k, h) = Σ_{n ∈ block_k} ψ_j(n) · λ(n) · λ(n+h)

and ψ_j is the Haar wavelet at scale 2^j.  The Haar coefficient at
block k of size 2^j is simply:

  C_j(k) = Σ_{n in first half} g(n) − Σ_{n in second half} g(n)

The pointwise bound tested:  max_k |C_j(k,h)| / 2^j  ≤  1/j²
If this holds for all j ≥ j₀, the wavelet-based Sobolev regularity
argument applies with effective c_MR ≈ 2 + exponent/2.

Full-scale parameters (from the paper):
  N = 4 × 10¹¹, h = 2, j = 16..37, 12,206,994 blocks
  64 threads, ~400 GB RAM

This script provides a small-scale Python implementation for
verification and testing.  The full-scale computation was performed
in C with OpenMP (not included in this repository).

Usage:
  python3 wavelet_verification.py                # default test mode
  python3 wavelet_verification.py --test         # quick test (N=2^20)
  python3 wavelet_verification.py --nmax 1000000000 --jmin 10 --jmax 30

Requirements: numpy, matplotlib (optional for plots)
"""

import numpy as np
import argparse
import sys
import os
import time

# Add parent directory for common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.liouville_sieve import compute_liouville


def wavelet_analysis(lam, N, h, j_min, j_max, verbose=True):
    """
    Compute Haar wavelet coefficients of g(n) = λ(n)·λ(n+h) at each
    dyadic scale j.

    Parameters
    ----------
    lam : int8 array
        Liouville function values λ[0..N+h].
    N : int
        Range upper bound.
    h : int
        Shift parameter.
    j_min, j_max : int
        Range of dyadic scales (block size = 2^j).

    Returns
    -------
    results : list of dicts with keys:
        j, block_size, n_blocks, max_C_norm, threshold, ratio, pw_pass,
        mean_sq_norm
    """
    results = []
    g = (lam[1:N + 1] * lam[1 + h:N + 1 + h]).astype(np.int64)

    for j in range(j_min, j_max + 1):
        block_size = 1 << j
        if block_size > N:
            break

        n_blocks = N // block_size
        if n_blocks == 0:
            break

        half = block_size // 2
        t0 = time.time()

        # Compute Haar coefficients for all blocks
        # C_j(k) = Σ_{first half} g(n) − Σ_{second half} g(n)
        # Use cumulative sums for efficiency
        usable = n_blocks * block_size
        g_blocks = g[:usable].reshape(n_blocks, block_size)
        first_half = g_blocks[:, :half].sum(axis=1)
        second_half = g_blocks[:, half:].sum(axis=1)
        C_j = first_half - second_half

        max_C = np.max(np.abs(C_j))
        max_C_norm = max_C / block_size  # |C_j|/2^j
        threshold = 1.0 / (j * j)       # 1/j²
        ratio = max_C_norm / threshold if threshold > 0 else float('inf')
        pw_pass = (max_C_norm <= threshold)

        # Mean square: MS = (1/K) Σ_k (C_j(k)/2^j)²
        ms = np.mean((C_j.astype(np.float64) / block_size) ** 2)
        # Normalised: MS / (1/4^j) — but we report MS / 4^j directly
        ms_norm = ms  # this is already |C|²/4^j averaged

        elapsed = time.time() - t0

        row = {
            'j': j,
            'block_size': block_size,
            'n_blocks': n_blocks,
            'max_C_norm': max_C_norm,
            'threshold': threshold,
            'ratio': ratio,
            'pw_pass': pw_pass,
            'mean_sq_norm': ms_norm,
            'elapsed': elapsed,
        }
        results.append(row)

        if verbose:
            flag = 'Y' if pw_pass else 'N'
            print(f"  j={j:3d}  K={n_blocks:>12,d}  "
                  f"max|C|/2^j={max_C_norm:.5f}  "
                  f"1/j²={threshold:.5f}  "
                  f"ratio={ratio:.3f}  PW={flag}  "
                  f"MS/4^j={ms_norm:.2e}  "
                  f"({elapsed:.1f}s)")

    return results


def print_table(results):
    """Print results in the format matching the paper's Table (§A.4)."""
    print(f"\n{'j':>4s}  {'K':>12s}  {'max|C|/2^j':>12s}  "
          f"{'1/j²':>8s}  {'Ratio':>7s}  {'PW':>3s}  {'MS/4^j':>12s}")
    print("─" * 70)
    for r in results:
        flag = 'Y' if r['pw_pass'] else 'N'
        print(f"{r['j']:4d}  {r['n_blocks']:12,d}  "
              f"{r['max_C_norm']:12.5f}  "
              f"{r['threshold']:8.5f}  "
              f"{r['ratio']:7.3f}  "
              f"{flag:>3s}  "
              f"{r['mean_sq_norm']:12.2e}")

    # Summary
    pw_scales = [r['j'] for r in results if r['pw_pass']]
    npw_scales = [r['j'] for r in results if not r['pw_pass']]
    if pw_scales:
        print(f"\nPointwise bound holds for j ≥ {min(pw_scales)} "
              f"({len(pw_scales)} scales)")
    if npw_scales:
        print(f"Fails at j ∈ {npw_scales}")

    # Mean-square power-law fit: MS(j) ~ j^{-β}
    js = np.array([r['j'] for r in results if r['mean_sq_norm'] > 0],
                  dtype=float)
    ms = np.array([r['mean_sq_norm'] for r in results
                   if r['mean_sq_norm'] > 0])
    if len(js) >= 3:
        coeffs = np.polyfit(np.log(js), np.log(ms), 1)
        beta = -coeffs[0]
        print(f"\nMS power-law fit: MS(j) ~ j^{{-{beta:.2f}}}")
        print(f"Effective c_MR = {beta / 2:.2f}")
        if beta / 2 >= 2:
            print(f"  → Exceeds Goldbach threshold (c_MR ≥ 2) ✓")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 4: Wavelet verification of pointwise '
                    'cancellation in λ(n)λ(n+h)')
    parser.add_argument('--nmax', type=int, default=None,
                        help='Range upper bound N (default: 2^20 in test, '
                             '10^8 otherwise)')
    parser.add_argument('--h', type=int, default=2,
                        help='Shift parameter (default: 2)')
    parser.add_argument('--jmin', type=int, default=10,
                        help='Minimum wavelet scale (default: 10)')
    parser.add_argument('--jmax', type=int, default=30,
                        help='Maximum wavelet scale (default: 30)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (N=2^20, j=5..15, <10s)')
    args = parser.parse_args()

    if args.test:
        N = 1 << 20  # ~10^6
        j_min, j_max = 5, 15
        h = args.h
        print("=" * 60)
        print(" WAVELET VERIFICATION — TEST MODE")
        print(f" N = {N:,}, h = {h}, j = {j_min}..{j_max}")
        print("=" * 60)
    else:
        N = args.nmax if args.nmax else 10**8
        j_min = args.jmin
        j_max = args.jmax
        h = args.h
        print("=" * 60)
        print(" EXPERIMENT 4: Wavelet Verification")
        print(f" N = {N:,}, h = {h}, j = {j_min}..{j_max}")
        print("=" * 60)

    print(f"\nSieving λ(n) for n ≤ {N + h:,}...")
    t0 = time.time()
    lam = compute_liouville(N + h, verbose=True)
    print(f"Sieve time: {time.time() - t0:.1f}s\n")

    print("Computing wavelet coefficients...")
    results = wavelet_analysis(lam, N, h, j_min, j_max)
    print_table(results)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
