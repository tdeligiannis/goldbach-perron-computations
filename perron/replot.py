#!/usr/bin/env python3
"""Regenerate plots from saved data with corrected titles.

Usage:
  python3 replot.py [--datadir results] [--outdir results]

Requires: pip install numpy matplotlib
"""

import numpy as np
import argparse
import os

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


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate Perron plots from saved data')
    parser.add_argument('--datadir', type=str, default='results',
                        help='Directory containing perron_data.npz')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    datafile = os.path.join(args.datadir, 'perron_data.npz')

    print(f"Loading data from {datafile}...")
    data = np.load(datafile, allow_pickle=True)
    scales = data['scales']
    t_values = data['t_values']

    # Reconstruct results
    all_results = {}
    for M in scales:
        exp = int(np.log10(M))
        k = f"M{exp}"
        all_results[M] = {
            'Ga': dict(zip(t_values, data[f"{k}_Ga"])),
            'Ga_prime': dict(zip(t_values, data[f"{k}_Ga_prime"])),
            'Ga_at_1': float(data[f"{k}_Ga_at_1"]),
            'S_over_M': float(data[f"{k}_S"]) / M,
        }

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(scales)))
    final_val = abs(all_results[scales[-1]]['Ga_prime'][t_values[-1]])

    # ── Plot 1: |G_a'(1+it)| vs t ──

    fig, ax = plt.subplots()
    for i, M in enumerate(scales):
        gp = np.array([abs(all_results[M]['Ga_prime'][t]) for t in t_values])
        exp = int(np.log10(M))
        ax.semilogx(t_values, gp, 'o-', color=colors[i], markersize=4,
                     label=f'$M = 10^{{{exp}}}$')

    ax.axhline(y=final_val, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'$\\approx {final_val:.3f}$')
    ax.set_xlabel('$t$', fontsize=14)
    ax.set_ylabel("$|G_a'(1+it)|$", fontsize=14)
    ax.set_title("Boundedness of $G_a'(1+it)$ at $\\sigma=1$\n"
                 u"(constant \u21d2 $W^{2,2}$ Sobolev regularity confirmed)",
                 fontsize=13)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(bottom=1.0, top=1.45)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "plot1_Ga_prime_bounded.png")
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p1}")
    plt.close(fig)

    # ── Plot 2: |H(t)|/t vs t ──

    fig, ax = plt.subplots()
    for i, M in enumerate(scales):
        Ga1 = all_results[M]['Ga_at_1']
        lip = np.array([abs(all_results[M]['Ga'][t] - Ga1) / t
                        for t in t_values])
        exp = int(np.log10(M))
        ax.semilogx(t_values, lip, 'o-', color=colors[i], markersize=4,
                     label=f'$M = 10^{{{exp}}}$')

    ax.axhline(y=final_val, color='red', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f"$|G_a'(1)| \\approx {final_val:.3f}$")
    ax.set_xlabel('$t$', fontsize=14)
    ax.set_ylabel("$|H(t)|/t = |G_a(1+it) - G_a(1)|\\, /\\, t$", fontsize=14)
    ax.set_title("Lipschitz test: $|H(t)|/t \\to |G_a'(1)|$\n"
                 u"(convergence \u21d2 Lipschitz; "
                 u"divergence \u21d2 only H\u00f6lder-$1/2$)",
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(bottom=0.8, top=1.5)
    fig.tight_layout()
    p2 = os.path.join(args.outdir, "plot2_lipschitz_test.png")
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p2}")
    plt.close(fig)

    # ── Plot 3: Cesàro ratio vs M ──

    fig, ax = plt.subplots()
    M_arr = np.array(scales, dtype=float)
    ratio_arr = np.array([
        abs(all_results[M]['S_over_M']) * np.log(M)**2
        for M in scales
    ])

    ax.semilogx(M_arr, ratio_arr, 'ko-', markersize=8, linewidth=2.5,
                label="$|S(M)/M| \\cdot (\\log M)^2$")
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.8,
               alpha=0.8, label="Goldbach threshold ($=1$)")
    ax.fill_between([M_arr[0] * 0.3, M_arr[-1] * 3], 0, 1, alpha=0.06,
                    color='green')

    for j, M in enumerate(scales):
        ax.annotate(f'{ratio_arr[j]:.4f}',
                    xy=(M_arr[j], ratio_arr[j]),
                    xytext=(0, 14), textcoords='offset points',
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_xlabel('$M$', fontsize=14)
    ax.set_ylabel("$|S(M)/M| \\cdot (\\log M)^2$", fontsize=14)
    ax.set_title(u"Ces\u00e0ro ratio vs Goldbach threshold\n"
                 u"(below red line \u21d2 $c \\geq 2$)",
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(bottom=0, top=max(0.3, max(ratio_arr) * 1.5))
    ax.set_xlim(M_arr[0] * 0.3, M_arr[-1] * 3)
    fig.tight_layout()
    p3 = os.path.join(args.outdir, "plot3_cesaro_ratio.png")
    fig.savefig(p3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {p3}")
    plt.close(fig)

    print("\nAll three plots regenerated with corrected titles (W^{2,2}).")


if __name__ == "__main__":
    main()
