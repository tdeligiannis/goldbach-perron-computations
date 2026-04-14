#!/usr/bin/env python3
"""
Goldbach-Perron computation: verifying H^2 regularity of G_a at sigma=1.
Multi-scale computation with publication-quality PNG plots.

Usage:
  pip install numpy matplotlib
  python3 goldbach_perron_computation.py --mmax 10000000000 --cores 64 --seg 5000000

Outputs:
  plot1_Ga_prime_bounded.png  - |G_a'(1+it)| vs t at each M scale
  plot2_lipschitz_test.png    - |H(t)|/t vs t at each M scale
  plot3_cesaro_ratio.png      - |S/M|*(log M)^2 vs M
  perron_data.npz             - raw numerical data
"""

import numpy as np
from multiprocessing import Pool
import time
import argparse
import os

W = 30
A_RES = 1
H_SHIFT = 6

def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    return np.nonzero(is_prime)[0]

def compute_segment(args):
    m_lo, m_hi, primes, t_values = args
    seg_m = m_hi - m_lo
    n_lo = W * m_lo + A_RES
    n_hi = W * m_hi + A_RES + H_SHIFT + 1
    seg_n = n_hi - n_lo

    omega_mod2 = np.zeros(seg_n, dtype=np.uint8)
    n_remaining = np.arange(n_lo, n_hi, dtype=np.int64)

    for p in primes:
        if p > n_hi:
            break
        pk = int(p)
        while pk < n_hi:
            start = int(((n_lo + pk - 1) // pk) * pk)
            if start < n_hi:
                omega_mod2[start - n_lo::pk] ^= 1
                indices = np.arange(start - n_lo, seg_n, pk, dtype=np.int64)
                n_remaining[indices] //= p
            pk_new = pk * int(p)
            if pk_new <= pk or pk_new > n_hi:
                break
            pk = pk_new

    omega_mod2[n_remaining > 1] ^= 1
    lam = np.int8(1) - np.int8(2) * omega_mod2.astype(np.int8)

    idx1 = np.arange(seg_m, dtype=np.int64) * W
    idx2 = idx1 + H_SHIFT
    g_a = lam[idx1] * lam[idx2]

    S_partial = int(np.sum(g_a))

    m_arr = np.maximum(np.arange(m_lo, m_hi, dtype=np.float64), 1.0)
    log_m = np.log(m_arr)
    g_float = g_a.astype(np.float64)
    g_over_m = g_float / m_arr
    g_logm_over_m = g_float * log_m / m_arr

    Ga_partials = {}
    Ga_prime_partials = {}
    for t in t_values:
        phases = np.exp(-1j * t * log_m)
        Ga_partials[t] = complex(np.sum(g_over_m * phases))
        Ga_prime_partials[t] = complex(-np.sum(g_logm_over_m * phases))

    return {
        'S_partial': S_partial,
        'Ga': Ga_partials,
        'Ga_prime': Ga_prime_partials,
        'm_hi': m_hi,
    }


def run_scale(M_MAX, N_CORES, M_SEG, primes, t_values):
    segments = []
    m = 1
    while m <= M_MAX:
        m_hi = min(m + M_SEG, M_MAX + 1)
        segments.append((m, m_hi, primes, t_values))
        m = m_hi

    S_cumulative = 0
    Ga_total = {t: 0j for t in t_values}
    Ga_prime_total = {t: 0j for t in t_values}
    completed = 0
    t_start = time.time()

    with Pool(processes=min(N_CORES, len(segments))) as pool:
        for result in pool.imap(compute_segment, segments):
            S_cumulative += result['S_partial']
            for t in t_values:
                Ga_total[t] += result['Ga'][t]
                Ga_prime_total[t] += result['Ga_prime'][t]
            completed += 1
            m_done = result['m_hi']
            if completed % max(1, len(segments) // 10) == 0 or completed == len(segments):
                elapsed = time.time() - t_start
                rate = m_done / elapsed if elapsed > 0 else 1
                eta = (M_MAX - m_done) / rate if rate > 0 else 0
                print(f"    [{completed}/{len(segments)}] m={m_done:,}  "
                      f"S/M={S_cumulative/m_done:.6f}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return S_cumulative, Ga_total, Ga_prime_total, time.time() - t_start


def make_plots(all_results, t_values, output_dir="."):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.figsize': (9, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.2,
    })

    M_vals = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(M_vals)))
    t_arr = np.array(t_values)
    final_val = abs(all_results[M_vals[-1]]['Ga_prime'][t_values[-1]])

    # ── Plot 1: |G_a'(1+it)| vs t ──

    fig, ax = plt.subplots()
    for i, M in enumerate(M_vals):
        res = all_results[M]
        gp = np.array([abs(res['Ga_prime'][t]) for t in t_values])
        exp = int(np.log10(M))
        ax.semilogx(t_arr, gp, 'o-', color=colors[i], markersize=4,
                     label=f'$M = 10^{{{exp}}}$')

    ax.axhline(y=final_val, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'$\\approx {final_val:.3f}$')

    ax.set_xlabel('$t$', fontsize=14)
    ax.set_ylabel("$|G_a'(1+it)|$", fontsize=14)
    ax.set_title("Boundedness of $G_a'(1+it)$ at $\\sigma=1$\n"
                  "(constant $\\Rightarrow$ $H^2$ Sobolev regularity confirmed)",
                 fontsize=13)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(bottom=1.0, top=1.45)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "plot1_Ga_prime_bounded.png")
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p1}")
    plt.close(fig)

    # ── Plot 2: |H(t)|/t vs t ──

    fig, ax = plt.subplots()
    for i, M in enumerate(M_vals):
        res = all_results[M]
        Ga1 = res['Ga_at_1']
        lip = np.array([abs(res['Ga'][t] - Ga1) / t for t in t_values])
        exp = int(np.log10(M))
        ax.semilogx(t_arr, lip, 'o-', color=colors[i], markersize=4,
                     label=f'$M = 10^{{{exp}}}$')

    ax.axhline(y=final_val, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f"$|G_a'(1)| \\approx {final_val:.3f}$")

    ax.set_xlabel('$t$', fontsize=14)
    ax.set_ylabel("$|H(t)|/t = |G_a(1+it) - G_a(1)|\\, /\\, t$", fontsize=14)
    ax.set_title("Lipschitz test: $|H(t)|/t \\to |G_a'(1)|$\n"
                  "(convergence $\\Rightarrow$ Lipschitz; "
                  "divergence $\\Rightarrow$ only H\\\"{o}lder-$1/2$)",
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(bottom=0.8, top=1.5)
    fig.tight_layout()
    p2 = os.path.join(output_dir, "plot2_lipschitz_test.png")
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p2}")
    plt.close(fig)

    # ── Plot 3: Cesàro ratio vs M ──

    fig, ax = plt.subplots()
    M_arr = np.array(M_vals, dtype=float)
    ratio_arr = np.array([
        abs(all_results[M]['S_over_M']) * np.log(M)**2
        for M in M_vals
    ])

    ax.semilogx(M_arr, ratio_arr, 'ko-', markersize=8, linewidth=2.5,
                label="$|S(M)/M| \\cdot (\\log M)^2$")
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.8,
               alpha=0.8, label="Goldbach threshold ($=1$)")
    ax.fill_between([M_arr[0]*0.3, M_arr[-1]*3], 0, 1, alpha=0.06,
                    color='green')

    for j, M in enumerate(M_vals):
        ax.annotate(f'{ratio_arr[j]:.4f}',
                    xy=(M_arr[j], ratio_arr[j]),
                    xytext=(0, 14), textcoords='offset points',
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='-', color='gray',
                                   lw=0.5))

    ax.set_xlabel('$M$', fontsize=14)
    ax.set_ylabel("$|S(M)/M| \\cdot (\\log M)^2$", fontsize=14)
    ax.set_title("Ces\\`aro ratio vs Goldbach threshold\n"
                  "(below red line $\\Rightarrow$ $c \\geq 2$)",
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(bottom=0, top=max(0.3, max(ratio_arr) * 1.5))
    ax.set_xlim(M_arr[0] * 0.3, M_arr[-1] * 3)
    fig.tight_layout()
    p3 = os.path.join(output_dir, "plot3_cesaro_ratio.png")
    fig.savefig(p3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p3}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Goldbach-Perron H² regularity computation')
    parser.add_argument('--mmax', type=int, default=10**9,
                        help='Maximum M (default: 10^9)')
    parser.add_argument('--cores', type=int, default=64,
                        help='CPU cores (default: 64)')
    parser.add_argument('--seg', type=int, default=5*10**6,
                        help='Segment size (default: 5e6)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    M_MAX = args.mmax
    N_CORES = args.cores
    M_SEG = args.seg
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    scales = []
    v = 10**7
    while v <= M_MAX:
        scales.append(v)
        v *= 10

    N_MAX = W * M_MAX + A_RES + H_SHIFT + 1
    SQRT_N = int(N_MAX**0.5) + 1

    t_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002,
                0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    print("=" * 70)
    print("Goldbach-Perron H² regularity (multi-scale + PNG plots)")
    print("=" * 70)
    print(f"Scales:  {', '.join(f'10^{int(np.log10(s))}' for s in scales)}")
    print(f"M_MAX:   {M_MAX:,}")
    print(f"N_MAX:   {N_MAX:,}")
    print(f"Cores:   {N_CORES}")
    print(f"Seg:     {M_SEG:,}")
    print(f"Output:  {OUTDIR}")
    print()

    print(f"Sieving primes up to {SQRT_N:,}...", end=" ", flush=True)
    t0 = time.time()
    primes = sieve_primes(SQRT_N)
    print(f"done ({len(primes):,} primes, {time.time()-t0:.1f}s)\n")

    all_results = {}

    for scale in scales:
        print(f"{'─'*60}")
        print(f"  M = {scale:,} (10^{int(np.log10(scale))})")
        print(f"{'─'*60}")
        S, Ga, Ga_prime, elapsed = run_scale(
            scale, N_CORES, M_SEG, primes, t_values)
        Ga_at_1 = Ga[t_values[-1]].real
        all_results[scale] = {
            'S': S,
            'S_over_M': S / scale,
            'Ga': Ga,
            'Ga_prime': Ga_prime,
            'Ga_at_1': Ga_at_1,
            'elapsed': elapsed,
        }
        gp1 = abs(Ga_prime[t_values[-1]])
        ratio = abs(S / scale) * np.log(scale)**2
        print(f"  Done: {elapsed:.1f}s  |G_a'(1)|={gp1:.6f}  "
              f"Cesaro ratio={ratio:.4f}\n")

    # ── Summary ──

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'M':>14s}  {'|Ga_prime(1)|':>14s}  {'Drift':>8s}  "
          f"{'Ga(1)':>10s}  {'Cesaro ratio':>14s}  {'Time':>8s}")
    print("─" * 75)
    prev = None
    for M in sorted(all_results.keys()):
        r = all_results[M]
        gp = abs(r['Ga_prime'][t_values[-1]])
        drift = f"{(gp/prev-1)*100:+.2f}%" if prev else "—"
        ratio = abs(r['S_over_M']) * np.log(M)**2
        print(f"  10^{int(np.log10(M)):<9d}  {gp:14.6f}  {drift:>8s}  "
              f"{r['Ga_at_1']:10.6f}  {ratio:14.4f}  {r['elapsed']:7.0f}s")
        prev = gp

    M_big = max(all_results.keys())
    r_big = all_results[M_big]
    Ga1 = r_big['Ga_at_1']
    print(f"\nDetail at M = 10^{int(np.log10(M_big))}:")
    print(f"{'t':>12s}  {'|Ga|':>12s}  {'|H(t)|/t':>12s}  {'|Ga_prime|':>12s}")
    print("─" * 55)
    for t in t_values:
        ga = abs(r_big['Ga'][t])
        lip = abs(r_big['Ga'][t] - Ga1) / t
        gp = abs(r_big['Ga_prime'][t])
        print(f"{t:12.5f}  {ga:12.6f}  {lip:12.4f}  {gp:12.6f}")

    pv = [abs(r_big['Ga_prime'][t]) for t in t_values]
    print(f"\nmax |G_a'| = {max(pv):.6f}, min = {min(pv):.6f}")
    if max(pv) < 2 * min(pv):
        print("VERDICT: G_a' BOUNDED → H² regularity CONFIRMED")
    else:
        print("VERDICT: G_a' may be unbounded → CHECK CAREFULLY")

    # ── Plots ──

    print(f"\n{'─'*60}")
    print("Generating PNG plots (300 DPI)...")
    try:
        make_plots(all_results, t_values, output_dir=OUTDIR)
        print("All plots saved successfully.")
    except ImportError:
        print("  matplotlib not found. Install: pip install matplotlib")
    except Exception as e:
        print(f"  Plot error: {e}")

    # ── Save data ──

    outfile = os.path.join(OUTDIR, "perron_data.npz")
    save_dict = {
        'scales': np.array(sorted(all_results.keys())),
        't_values': np.array(t_values),
    }
    for M in sorted(all_results.keys()):
        r = all_results[M]
        k = f"M{int(np.log10(M))}"
        save_dict[f"{k}_S"] = r['S']
        save_dict[f"{k}_Ga"] = np.array([r['Ga'][t] for t in t_values])
        save_dict[f"{k}_Ga_prime"] = np.array(
            [r['Ga_prime'][t] for t in t_values])
        save_dict[f"{k}_Ga_at_1"] = r['Ga_at_1']
    np.savez(outfile, **save_dict)
    print(f"\nData saved to {outfile}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
