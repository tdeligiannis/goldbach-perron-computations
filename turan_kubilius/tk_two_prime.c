/*
 * tk_two_prime.c — Experiment 2: Turán–Kubilius Two-Prime Interactions
 *
 * Paper: T. Deligiannis, "A reduction of binary Goldbach to a
 *        four-point Chowla bound via the polynomial squaring identity"
 *        (2026), Appendix A, §A.2.
 *
 * EXPERIMENT: Turán–Kubilius coherent cancellation in the four-point sum.
 *
 * Tests whether sum_p C_p * A_{7,p} shows genuine cancellation
 * beyond the incoherent bound sum_p |C_p| * |A_{7,p}|.
 *
 * For each prime p != 7, p > z0:
 *   C_p(n) = (-1)^{v_p(Q_k(n))}  where Q_k(n) = n(n+k)(n+h)(n+k+h)
 *   E_p    = (p-7)/(p+1)          (theoretical mean, from Prop 5.3)
 *   Delta_p(n) = C_p(n) - E_p_empirical
 *   A_{7,p}    = mean_n[ Delta_7(n) * Delta_p(n) ]
 *
 * Output: A_{7,p}, coherent partial sum, incoherent partial sum,
 *         cancellation ratio, as function of primes included.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o calc1_tk_coherent calc1_tk_coherent.c -lm
 *
 * Run:
 *   OMP_NUM_THREADS=64 ./calc1_tk_coherent
 *
 * Pop!_OS / 64-core / 540 GB RAM
 * Expected runtime: ~2–4 hours for X_MAX = 2e9 (conservative),
 *                   ~20 hours for X_MAX = 1e10
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/*  Configuration — edit these before running                          */
/* ------------------------------------------------------------------ */
#ifndef X_MAX
#define X_MAX       2000000000ULL
#endif
#define K           4               /* shift k in Q_k */
#define H           6               /* shift h in Q_k */
#define P_MAX_TK    300             /* primes p in [11, P_MAX_TK] for TK sum */
#define SEG_SIZE    (1ULL << 23)    /* 8 MB sieve segments */
#define MARGIN      64              /* extra headroom beyond X_MAX */

/* Checkpoint X values for tracking cancellation ratio over X */
static const uint64_t CHECKPOINTS[] = {
    100000000ULL,    /* 1e8 */
    200000000ULL,    /* 2e8 */
    500000000ULL,    /* 5e8 */
    1000000000ULL,   /* 1e9 */
    2000000000ULL,   /* 2e9 */
    5000000000ULL,   /* 5e9  -- only reached if X_MAX >= 5e9 */
    10000000000ULL,  /* 1e10 -- only reached if X_MAX >= 1e10 */
};
#define N_CHECKPOINTS (sizeof(CHECKPOINTS)/sizeof(CHECKPOINTS[0]))

/* ------------------------------------------------------------------ */
/*  Utilities                                                          */
/* ------------------------------------------------------------------ */
static double elapsed(struct timespec t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
}

/* Simple prime sieve for primes up to LIMIT (returns count) */
static uint32_t *sieve_primes(uint64_t limit, int *count) {
    uint64_t sz = limit + 1;
    uint8_t *is_comp = calloc(sz, 1);
    if (!is_comp) { perror("calloc primes"); exit(1); }
    is_comp[0] = is_comp[1] = 1;
    for (uint64_t i = 2; i * i <= limit; i++)
        if (!is_comp[i])
            for (uint64_t j = i*i; j <= limit; j += i)
                is_comp[j] = 1;

    /* Count primes */
    int n = 0;
    for (uint64_t i = 2; i <= limit; i++) if (!is_comp[i]) n++;
    uint32_t *primes = malloc((size_t)n * sizeof(uint32_t));
    if (!primes) { perror("malloc primes"); exit(1); }
    int idx = 0;
    for (uint64_t i = 2; i <= limit; i++) if (!is_comp[i]) primes[idx++] = (uint32_t)i;
    free(is_comp);
    *count = n;
    return primes;
}

/* ------------------------------------------------------------------ */
/*  Liouville parity sieve: lam[n] = Omega(n) mod 2                   */
/*  lambda(n) = 1 - 2*lam[n]                                          */
/*                                                                     */
/*  Strategy: parallel segmented sieve.                                */
/*  Each thread owns a disjoint segment of lam[].                      */
/*  For each segment, apply every prime p (and its powers p^k <= X).  */
/* ------------------------------------------------------------------ */
static uint8_t *build_liouville(uint64_t X, int num_primes, uint32_t *primes) {
    uint64_t sz = X + MARGIN + 1;
    uint8_t *lam = calloc(sz, 1);
    if (!lam) { perror("calloc lam"); exit(1); }

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    printf("  Sieving Liouville to X=%llu with %d primes, %d threads...\n",
           (unsigned long long)X, num_primes, omp_get_max_threads());
    fflush(stdout);

    /* Number of segments */
    uint64_t n_segs = (X + SEG_SIZE) / SEG_SIZE;

    #pragma omp parallel for schedule(dynamic, 4)
    for (uint64_t seg = 0; seg < n_segs; seg++) {
        uint64_t s = seg * SEG_SIZE;
        uint64_t e = s + SEG_SIZE - 1;
        if (e > X + MARGIN) e = X + MARGIN;

        for (int pi = 0; pi < num_primes; pi++) {
            uint64_t p = primes[pi];
            if (p > e) break;  /* primes are sorted; no more can contribute */

            uint64_t pk = p;
            while (pk <= e) {
                /* first multiple of pk in [s, e] */
                uint64_t first = (s == 0) ? pk : ((s + pk - 1) / pk) * pk;
                for (uint64_t m = first; m <= e; m += pk)
                    lam[m] ^= 1;

                /* next power; check overflow */
                if (pk > (X + MARGIN) / p) break;
                pk *= p;
            }
        }

        if (seg % 256 == 0) {
            #pragma omp critical
            {
                printf("  ...seg %llu/%llu (%.1f%%) elapsed=%.0fs\r",
                       (unsigned long long)seg, (unsigned long long)n_segs,
                       100.0*seg/n_segs, elapsed(t0));
                fflush(stdout);
            }
        }
    }

    printf("\n  Sieve done in %.1fs.\n", elapsed(t0));
    return lam;
}

/* ------------------------------------------------------------------ */
/*  Compute v_p parity array for prime p:                              */
/*  vp_par[n] = (v_p(n) + v_p(n+k) + v_p(n+h) + v_p(n+k+h)) mod 2  */
/*  for n = 1 .. X                                                     */
/* ------------------------------------------------------------------ */
static uint8_t *build_vp_parity(uint64_t X, uint64_t p,
                                 uint64_t k, uint64_t h) {
    uint64_t sz = X + k + h + MARGIN + 1;
    uint8_t *vp = calloc(sz, 1);
    if (!vp) { perror("calloc vp"); exit(1); }

    uint64_t pk = p;
    while (pk < sz) {
        for (uint64_t m = pk; m < sz; m += pk)
            vp[m] ^= 1;
        if (pk > sz / p) break;
        pk *= p;
    }

    /* C_p(n) = vp[n] ^ vp[n+k] ^ vp[n+h] ^ vp[n+k+h] stored in vp[n] */
    /* Reuse vp array: build combined parity into a new array */
    uint8_t *cp = malloc(X + 1);
    if (!cp) { perror("malloc cp"); exit(1); }

    for (uint64_t n = 1; n <= X; n++)
        cp[n] = vp[n] ^ vp[n+k] ^ vp[n+h] ^ vp[n+k+h];

    free(vp);
    return cp;   /* cp[n] = 0 or 1; C_p(n) = 1 - 2*cp[n] */
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(void) {
    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("========================================================\n");
    printf(" EXPERIMENT 1: TK Coherent Cancellation\n");
    printf(" X_MAX=%llu  k=%d  h=%d  P_MAX=%d  threads=%d\n",
           (unsigned long long)X_MAX, K, H, P_MAX_TK, omp_get_max_threads());
    printf("========================================================\n\n");

    /* ---- 1. Prime sieve up to X_MAX ---- */
    int n_primes;
    printf("Step 1: Prime sieve to %llu...\n", (unsigned long long)X_MAX);
    uint32_t *primes = sieve_primes(X_MAX + H + K + MARGIN, &n_primes);
    printf("  Found %d primes.\n\n", n_primes);

    /* ---- 2. Liouville sieve ---- */
    printf("Step 2: Liouville parity sieve...\n");
    uint8_t *lam = build_liouville(X_MAX, n_primes, primes);

    /* Quick sanity check: lambda(2)=-1, lambda(4)=1, lambda(6)=-1 */
    printf("  Sanity: lam[2]=%d lam[4]=%d lam[6]=%d (expect 1,0,1)\n\n",
           lam[2], lam[4], lam[6]);

    /* ---- 3. Build Delta_7 ---- */
    printf("Step 3: Building C_7 and Delta_7...\n");
    uint8_t *cp7 = build_vp_parity(X_MAX, 7, K, H);

    /* Compute empirical E_7 = mean(1 - 2*cp7[n]) */
    double sum_C7 = 0.0;
    #pragma omp parallel for reduction(+:sum_C7) schedule(static)
    for (uint64_t n = 1; n <= X_MAX; n++)
        sum_C7 += (double)(1 - 2*(int)cp7[n]);
    double E7_emp = sum_C7 / (double)X_MAX;
    printf("  E_7 empirical = %.8f (theory = 0)\n\n", E7_emp);
    /* Delta_7(n) = (1 - 2*cp7[n]) - E7_emp; stored implicitly */

    /* ---- 4. For each prime p != 7, compute A_{7,p} ---- */
    /* Find TK primes in [11, P_MAX_TK], excluding 7 */
    int n_tk = 0;
    uint32_t tk_primes[512];
    for (int i = 0; i < n_primes && primes[i] <= (uint32_t)P_MAX_TK; i++)
        if (primes[i] >= 11 && primes[i] != 7)
            tk_primes[n_tk++] = primes[i];
    printf("Step 4: Computing A_{7,p} for %d primes in [11, %d]...\n",
           n_tk, P_MAX_TK);

    double *A7p       = calloc(n_tk, sizeof(double));
    double *Ep_theory = calloc(n_tk, sizeof(double));
    double *Ep_emp_arr= calloc(n_tk, sizeof(double));

    for (int pi = 0; pi < n_tk; pi++) {
        uint64_t p = tk_primes[pi];
        Ep_theory[pi] = ((double)p - 7.0) / ((double)p + 1.0);

        uint8_t *cpp = build_vp_parity(X_MAX, p, K, H);

        /* Empirical E_p */
        double sum_Cp = 0.0;
        #pragma omp parallel for reduction(+:sum_Cp) schedule(static)
        for (uint64_t n = 1; n <= X_MAX; n++)
            sum_Cp += (double)(1 - 2*(int)cpp[n]);
        double Ep_emp = sum_Cp / (double)X_MAX;
        Ep_emp_arr[pi] = Ep_emp;

        /* A_{7,p} = mean_n[ (C7(n) - E7) * (Cp(n) - Ep) ] */
        double sum_cross = 0.0;
        #pragma omp parallel for reduction(+:sum_cross) schedule(static)
        for (uint64_t n = 1; n <= X_MAX; n++) {
            double d7  = (double)(1 - 2*(int)cp7[n]) - E7_emp;
            double dp  = (double)(1 - 2*(int)cpp[n]) - Ep_emp;
            sum_cross += d7 * dp;
        }
        A7p[pi] = sum_cross / (double)X_MAX;

        free(cpp);

        if ((pi+1) % 5 == 0 || pi == n_tk-1)
            printf("  [%d/%d] p=%u done, elapsed=%.1fs\n",
                   pi+1, n_tk, (uint32_t)p, elapsed(t_start));
        fflush(stdout);
    }

    /* ---- 5. Compute coefficient C_coeff_p = prod_{q in TK, q != p} E_q ---- */
    /* Use log-sum for numerical stability */
    double log_full_prod = 0.0;
    for (int pi = 0; pi < n_tk; pi++) {
        double ep = Ep_theory[pi];
        if (ep > 0) log_full_prod += log(ep);
        /* E_p near 0 (small p) → large negative log → product near 0 */
    }
    double full_prod = exp(log_full_prod);

    printf("\n  Product of E_p for p in [11, %d]: %.6e\n\n", P_MAX_TK, full_prod);

    /* ---- 6. Report ---- */
    printf("%-5s  %-8s  %-17s  %-6s  %-18s  %-18s  %-12s\n",
           "p", "E_p", "A_{7,p}", "sign", "partial_coh", "partial_incoh", "cancel_ratio");
    printf("%s\n", "--------------------------------------------------------------------------------------------");

    double coherent   = 0.0;
    double incoherent = 0.0;

    for (int pi = 0; pi < n_tk; pi++) {
        double ep = Ep_theory[pi];
        /* C_coeff_p = full_prod / ep (product excluding this prime) */
        double coeff = (ep > 1e-12) ? full_prod / ep : 0.0;
        double term  = coeff * A7p[pi];
        coherent   += term;
        incoherent += fabs(coeff) * fabs(A7p[pi]);
        double ratio = (incoherent > 0) ? fabs(coherent)/incoherent : 1.0;

        printf("%-5u  %-8.4f  %-+17.11f  %-6c  %-18.12f  %-18.12f  %-12.5f\n",
               tk_primes[pi], ep, A7p[pi],
               A7p[pi] >= 0 ? '+' : '-',
               coherent, incoherent, ratio);
    }

    printf("\nFinal cancellation ratio = |coherent|/|incoherent| = %.6f\n",
           (incoherent > 0) ? fabs(coherent)/incoherent : 1.0);

    int n_neg = 0;
    for (int pi = 0; pi < n_tk; pi++) if (A7p[pi] < 0) n_neg++;
    printf("Fraction of negative A_{7,p}: %.3f  (%d / %d)\n",
           (double)n_neg/n_tk, n_neg, n_tk);

    /* ---- 7. Checkpoint: cancellation ratio vs X ---- */
    printf("\n--- Cancellation ratio as function of X (first 5 primes) ---\n");
    printf("%-12s  %-18s  %-18s  %-12s\n",
           "X", "partial_coh", "partial_incoh", "cancel_ratio");

    /* Re-run with X = each checkpoint, using only first 20 TK primes for speed */
    int n_cp_primes = (n_tk < 20) ? n_tk : 20;
    for (int ci = 0; ci < (int)N_CHECKPOINTS; ci++) {
        uint64_t Xc = CHECKPOINTS[ci];
        if (Xc > X_MAX) break;

        double coh2 = 0.0, incoh2 = 0.0;
        for (int pi = 0; pi < n_cp_primes; pi++) {
            /* Re-sum A7p using only first Xc values */
            uint64_t p = tk_primes[pi];
            uint8_t *cpp = build_vp_parity(Xc, p, K, H);
            uint8_t *c7c = build_vp_parity(Xc, 7, K, H);
            double se7 = 0.0, sep = 0.0, scross = 0.0;
            #pragma omp parallel for reduction(+:se7,sep,scross) schedule(static)
            for (uint64_t n = 1; n <= Xc; n++) {
                se7    += (double)(1 - 2*(int)c7c[n]);
                sep    += (double)(1 - 2*(int)cpp[n]);
                scross += (double)(1-2*(int)c7c[n])*(double)(1-2*(int)cpp[n]);
            }
            double e7c  = se7  / (double)Xc;
            double epc  = sep  / (double)Xc;
            double a7pc = scross/(double)Xc - e7c*epc;
            double ep   = Ep_theory[pi];
            double coeff = (ep > 1e-12) ? full_prod/ep : 0.0;
            coh2   += coeff * a7pc;
            incoh2 += fabs(coeff)*fabs(a7pc);
            free(c7c); free(cpp);
        }
        double r = (incoh2 > 0) ? fabs(coh2)/incoh2 : 1.0;
        printf("%-12llu  %-18.12f  %-18.12f  %-12.6f\n",
               (unsigned long long)Xc, coh2, incoh2, r);
    }

    printf("\nTotal elapsed: %.1fs\n", elapsed(t_start));

    free(lam); free(cp7); free(primes);
    free(A7p); free(Ep_theory); free(Ep_emp_arr);
    return 0;
}
