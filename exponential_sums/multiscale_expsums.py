#!/usr/bin/env python3
"""
Experiment 5c: Multi-Scale Exponential Sum Decay
==================================================

Paper: T. Deligiannis, "A reduction of binary Goldbach to a
       four-point Chowla bound via the polynomial squaring identity"
       (2026), Appendix A, §A.5 (multi-scale exponential sums).

For g_k(n) = λ(n)·λ(n+k) and baseline λ(n), computes

  max_{(a,q)=1} |ĝ(a/q)| / X

at multiple scales X ∈ {10⁵, 10⁶, 10⁷, 10⁸} and fixed moduli q.

The Goldbach threshold requires |ĝ(a/q)|/X ≤ (log X)^{−(1+ε)}.
This script tests whether the binary correlation g_k equidistributes
in arithmetic progressions comparably to the multiplicative baseline λ.

Usage:
  python3 multiscale_expsums.py              # default (X up to 10^7)
  python3 multiscale_expsums.py --test       # quick test (X up to 10^5)
  python3 multiscale_expsums.py --xmax 8     # X up to 10^8

Requirements: numpy
"""

import numpy as np
from math import gcd, isqrt, log
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.liouville_sieve import compute_liouville


def coprime_mask(q):
    """Boolean mask: mask[a] = True iff gcd(a, q) = 1."""
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


def max_exp_sum_normed(g, q):
    """
    Compute max_{(a,q)=1} |ĝ(a/q)| / X where X = len(g).

    ĝ(a/q) = Σ_{r=0}^{q-1} S_r · e(ra/q), with S_r = Σ_{n≡r(q)} g(n).
    """
    X = len(g)
    full = (X // q) * q
    S = g[:full].reshape(-1, q).sum(axis=0, dtype=np.int64)
    S = np.roll(S, 1)  # S[r] = sum of g(n) for n ≡ r (mod q)
    for i in range(full, X):
        S[(i + 1) % q] += int(g[i])

    dft = np.fft.fft(S.astype(np.complex128))
    mags = np.abs(dft)
    mask = coprime_mask(q)
    coprime_a = np.where(mask)[0]
    if len(coprime_a) == 0:
        return 0.0
    return float(np.max(mags[coprime_a])) / X


def run_multiscale(lam, scales, shifts, q_values, verbose=True):
    """
    For each X in scales, compute max|ĝ(a/q)|/X at each q.

    Returns dict: results[key][X_val][q] = value
    where key is 'baseline' or shift k.
    """
    results = {}

    # Baseline: multiplicative λ
    if verbose:
        print("  Computing baseline λ(n)...")
    results['baseline'] = {}
    for X_val in scales:
        exp_data = {}
        g = lam[1:X_val + 1].copy()
        for q in q_values:
            if q >= X_val:
                continue
            exp_data[q] = max_exp_sum_normed(g, q)
        results['baseline'][X_val] = exp_data

    # Binary correlations g_k(n) = λ(n)·λ(n+k)
    for k in shifts:
        if verbose:
            print(f"  Computing k = {k}...")
        results[k] = {}
        for X_val in scales:
            if X_val + k >= len(lam):
                continue
            g = (lam[1:X_val + 1] * lam[1 + k:X_val + 1 + k]).astype(np.int8)
            exp_data = {}
            for q in q_values:
                if q >= X_val:
                    continue
                exp_data[q] = max_exp_sum_normed(g, q)
            results[k][X_val] = exp_data

    return results


def print_table(results, scales, shifts, q_values):
    """Print table matching the paper's format."""
    print(f"\n  max |ĝ(a/q)|/X  (should decrease with X)")
    header = f"  {'q':>5s}"
    for X_val in scales:
        header += f"  {'X=' + f'{X_val:.0e}':>10s}"
    header += f"  {'1/(lnX)^1.1':>12s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for q in q_values:
        # Baseline row
        line = f"  {q:5d} λ "
        for X_val in scales:
            v = results['baseline'].get(X_val, {}).get(q, float('nan'))
            line += f"  {v:10.6f}"
        ref = 1.0 / log(scales[-1]) ** 1.1
        line += f"  {ref:12.6f}"
        print(line)

        # Correlation rows
        for k in shifts:
            line = f"       k={k:<2d}"
            for X_val in scales:
                v = results.get(k, {}).get(X_val, {}).get(q, float('nan'))
                line += f"  {v:10.6f}"
            print(line)
        print()

    # Summary: slope in |ĝ|/X ~ X^{-α} by log-log regression
    print(f"\n  Scaling exponents (|ĝ|/X ~ X^{{-α}}):")
    for q in [31, 97, 499]:
        if q not in q_values:
            continue
        for key in ['baseline'] + list(shifts):
            label = 'λ' if key == 'baseline' else f'k={key}'
            xs = []
            vs = []
            for X_val in scales:
                v = results.get(key, {}).get(X_val, {}).get(q, None)
                if v is not None and v > 0:
                    xs.append(np.log(X_val))
                    vs.append(np.log(v))
            if len(xs) >= 2:
                slope = np.polyfit(xs, vs, 1)[0]
                print(f"    q={q:3d}, {label:>8s}: α = {-slope:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5c: Multi-scale exponential sum decay')
    parser.add_argument('--xmax', type=int, default=7,
                        help='Maximum scale as exponent (10^xmax, default: 7)')
    parser.add_argument('--shifts', type=int, nargs='+', default=[1, 2, 3, 6],
                        help='Shift values k (default: 1 2 3 6)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (X up to 10^5, <10s)')
    args = parser.parse_args()

    if args.test:
        xmax = 5
        shifts = [1, 3]
        q_values = [7, 11, 31, 97]
    else:
        xmax = args.xmax
        shifts = args.shifts
        q_values = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 67, 71, 79, 83, 89, 97, 101, 127, 151,
                    197, 211, 251, 307, 401, 499]

    scales = [10 ** e for e in range(5, xmax + 1)]
    max_shift = max(shifts)
    N_sieve = max(scales) + max_shift + 1

    print("=" * 60)
    print(" EXPERIMENT 5c: Multi-Scale Exponential Sum Decay")
    print(f" Scales: {', '.join(f'10^{e}' for e in range(5, xmax + 1))}")
    print(f" Shifts: {shifts}")
    print(f" Moduli: {len(q_values)} values, q ≤ {max(q_values)}")
    print("=" * 60)

    t0 = time.time()
    lam = compute_liouville(N_sieve, verbose=True)

    results = run_multiscale(lam, scales, shifts, q_values)
    print_table(results, scales, shifts, q_values)

    # Goldbach check
    ref_11 = 1.0 / log(max(scales)) ** 1.1
    ref_2 = 1.0 / log(max(scales)) ** 2
    print(f"\n  Goldbach threshold at X = {max(scales):.0e}:")
    print(f"    1/(ln X)^{{1.1}} = {ref_11:.6f}")
    print(f"    1/(ln X)^2     = {ref_2:.6f}")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
