/*
 * c4_covariance.c — Experiment 1: Four-Point Covariance Splitting
 *
 * Paper: T. Deligiannis, "A reduction of binary Goldbach to a
 *        four-point Chowla bound via the polynomial squaring identity"
 *        (2026), Appendix A, §A.1.
 *
 * EXPERIMENT: C4 covariance splitting and decay exponent measurement.
 *
 * For each checkpoint X in [1e8, ..., 1e10], compute:
 *   C4(k)   = (1/X) sum_{n<=X} lambda(n)*lambda(n+k)*lambda(n+h)*lambda(n+k+h)
 *   mu1     = (1/X) sum_{n<=X} lambda(n)*lambda(n+k)
 *   mu2     = (1/X) sum_{n<=X} lambda(n+h)*lambda(n+k+h)
 *   Cov     = C4 - mu1*mu2
 *
 * Across multiple shifts k and the fixed shift h=6.
 *
 * The key question: does |Cov| ~ (log X)^{-alpha} with alpha > 2?
 * Does |mu1*mu2| remain negligible relative to |Cov|?
 *
 * Also computes the partial sum S2(X) = sum_{n<=X} lambda(n)*lambda(n+k)
 * and fits its decay exponent.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o calc3_c4_covariance calc3_c4_covariance.c -lm
 *
 * Run:
 *   OMP_NUM_THREADS=64 ./calc3_c4_covariance
 *
 * Pop!_OS / 64-core / 540 GB RAM
 * Expected runtime: ~1–2 hours for X_MAX = 1e10
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/*  Configuration                                                      */
/* ------------------------------------------------------------------ */
#ifndef X_MAX
#define X_MAX       10000000000ULL
#endif
#define H           6               /* fixed Goldbach shift              */
#define SEG_SIZE    (1ULL << 23)    /* 8 MB segments                    */
#define MARGIN      64

/* Shifts k to test — ODD PRIMES only, gcd(k,h=6)=1, so avoid k=3 */
static const int K_VALS[]  = {11, 13, 17, 19, 23, 29, 31};
#define N_K_VALS (sizeof(K_VALS)/sizeof(K_VALS[0]))

/* Checkpoint X values */
static const uint64_t CP[] = {
    10000000ULL,     /* 1e7  */
    30000000ULL,     /* 3e7  */
    100000000ULL,    /* 1e8  */
    300000000ULL,    /* 3e8  */
    1000000000ULL,   /* 1e9  */
    3000000000ULL,   /* 3e9  */
    10000000000ULL,  /* 1e10 */
};
#define N_CP (sizeof(CP)/sizeof(CP[0]))

/* ------------------------------------------------------------------ */
static double elapsed_s(struct timespec t0) {
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec-t0.tv_sec)+1e-9*(t1.tv_nsec-t0.tv_nsec);
}

/* ------------------------------------------------------------------ */
/*  Prime sieve (returns list of primes up to limit)                  */
/* ------------------------------------------------------------------ */
static uint32_t *prime_list(uint64_t limit, int *cnt) {
    uint8_t *c = calloc(limit+1, 1);
    if (!c) { perror("prime_list"); exit(1); }
    c[0]=c[1]=1;
    for (uint64_t i=2; i*i<=limit; i++)
        if (!c[i]) for (uint64_t j=i*i; j<=limit; j+=i) c[j]=1;
    int n=0;
    for (uint64_t i=2; i<=limit; i++) if (!c[i]) n++;
    uint32_t *p = malloc((size_t)n*4); if (!p){perror("prime_list");exit(1);}
    int idx=0;
    for (uint64_t i=2; i<=limit; i++) if (!c[i]) p[idx++]=(uint32_t)i;
    free(c); *cnt=n; return p;
}

/* ------------------------------------------------------------------ */
/*  Liouville parity sieve (parallel segmented)                       */
/*  lam[n] = Omega(n) % 2;  lambda(n) = 1 - 2*lam[n]                */
/* ------------------------------------------------------------------ */
static uint8_t *build_lam(uint64_t X, int np, uint32_t *primes) {
    uint64_t sz = X + MARGIN + 1;
    uint8_t *lam = calloc(sz, 1);
    if (!lam) { perror("build_lam"); exit(1); }

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    printf("  Sieve: X=%llu, threads=%d ...\n",
           (unsigned long long)X, omp_get_max_threads());
    fflush(stdout);

    uint64_t n_segs = (X + MARGIN + SEG_SIZE) / SEG_SIZE;

    #pragma omp parallel for schedule(dynamic, 8)
    for (uint64_t seg = 0; seg < n_segs; seg++) {
        uint64_t s = seg * SEG_SIZE;
        uint64_t e = s + SEG_SIZE - 1;
        if (e >= sz) e = sz - 1;

        for (int pi = 0; pi < np; pi++) {
            uint64_t p = primes[pi];
            if (p > e) break;
            uint64_t pk = p;
            while (pk <= e) {
                uint64_t first = (s < pk) ? pk : ((s+pk-1)/pk)*pk;
                for (uint64_t m = first; m <= e; m += pk)
                    lam[m] ^= 1;
                if (pk > (sz-1)/p) break;
                pk *= p;
            }
        }

        if (seg % 512 == 0) {
            #pragma omp critical
            { printf("  ...%.1f%%  %.0fs\r", 100.0*seg/n_segs, elapsed_s(t0));
              fflush(stdout); }
        }
    }
    printf("\n  Sieve done: %.1fs\n", elapsed_s(t0));
    return lam;
}

/* ------------------------------------------------------------------ */
/*  Compute running statistics up to X_MAX for shift k                */
/*                                                                     */
/*  Accumulates in large int64 sums, snapshots at each checkpoint.    */
/*  Uses lam[] array built once.                                       */
/* ------------------------------------------------------------------ */
static void compute_stats(const uint8_t *lam, int k,
                           double *out_C4, double *out_mu1,
                           double *out_mu2, double *out_Cov,
                           double *out_S2) {
    /*
     * We need:
     *   sum4  = sum_{n=1}^{X} lp1[n] * lp2[n]   (for C4)
     *   sum1  = sum_{n=1}^{X} lp1[n]              (for mu1)
     *   sum2  = sum_{n=1}^{X} lp2[n]              (for mu2)
     *   sum2k = sum_{n=1}^{X} lp1[n]              (same as sum1, for 2-pt)
     *
     * where lp1[n] = lambda(n)*lambda(n+k) = (1-2*lam[n])*(1-2*lam[n+k])
     *               = 1 - 2*lam[n] - 2*lam[n+k] + 4*lam[n]*lam[n+k]
     *               = 1 - 2*(lam[n] ^ lam[n+k])   [since lam values are 0/1]
     *
     * Similarly lp2[n] = 1 - 2*(lam[n+H] ^ lam[n+k+H])
     * lp1*lp2 = (1-2*A)*(1-2*B) = 1 - 2*A - 2*B + 4*A*B
     *         = 1 - 2*(A ^ B) - (sign correction)
     * Wait, lp1 and lp2 are +/-1 valued.
     * lp1*lp2 = (1-2*a)*(1-2*b) where a,b in {0,1}.
     *         = 1 - 2*a - 2*b + 4*a*b
     *         = 1 - 2*(a XOR b) - 2*a*b*(-2) ... let's be explicit:
     * (a,b)=(0,0): 1; (0,1)=-1; (1,0)=-1; (1,1)=+1
     * = 1 - 2*(a^b) ... but (1,1): a^b=0 gives 1 ✓; (0,1): a^b=1 gives -1 ✓. Yes!
     * So lp1*lp2 = 1 - 2*(a^b).
     *
     * Use bit arithmetic: a = lam[n]^lam[n+k], b = lam[n+H]^lam[n+k+H]
     * lp1*lp2 parity bit = a^b = lam[n]^lam[n+k]^lam[n+H]^lam[n+k+H]
     */
    int kk = k, hh = H;

    for (int ci = 0; ci < (int)N_CP; ci++) {
        uint64_t Xc = CP[ci];
        if (Xc > X_MAX) break;

        int64_t s4=0, s1=0, s2=0;

        #pragma omp parallel reduction(+:s4,s1,s2)
        {
            int64_t ls4=0, ls1=0, ls2=0;
            #pragma omp for schedule(static)
            for (uint64_t n = 1; n <= Xc; n++) {
                uint8_t a  = lam[n]    ^ lam[n+kk];
                uint8_t b  = lam[n+hh] ^ lam[n+kk+hh];
                ls4 += (int)(1 - 2*(int)(a ^ b));
                ls1 += (int)(1 - 2*(int)a);
                ls2 += (int)(1 - 2*(int)b);
            }
            s4 += ls4; s1 += ls1; s2 += ls2;
        }

        double X = (double)Xc;
        double C4  = (double)s4 / X;
        double mu1 = (double)s1 / X;
        double mu2 = (double)s2 / X;
        double Cov = C4 - mu1*mu2;
        double S2  = (double)s1 / X;  /* same as mu1 for 2-pt sum */

        out_C4 [ci] = C4;
        out_mu1[ci] = mu1;
        out_mu2[ci] = mu2;
        out_Cov[ci] = Cov;
        out_S2 [ci] = S2;
    }
}

/* ------------------------------------------------------------------ */
/*  Fit power-law decay: |val| ~ (log X)^{-alpha}                     */
/*  Returns alpha (positive means decay).                              */
/* ------------------------------------------------------------------ */
static double fit_alpha(const double *vals, const uint64_t *xs, int n) {
    /* OLS of log|val| ~ alpha * log(log X) + const */
    double sx=0, sy=0, sxx=0, sxy=0;
    int cnt=0;
    for (int i=0; i<n; i++) {
        if (xs[i] > X_MAX) break;
        double v = fabs(vals[i]);
        if (v < 1e-12) continue;
        double x = log(log((double)xs[i]));
        double y = log(v);
        sx  += x; sy  += y;
        sxx += x*x; sxy += x*y;
        cnt++;
    }
    if (cnt < 2) return 0.0;
    double denom = cnt*sxx - sx*sx;
    if (fabs(denom) < 1e-12) return 0.0;
    double slope = (cnt*sxy - sx*sy) / denom;
    return -slope;  /* positive => decay */
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(void) {
    struct timespec t_start; clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("==========================================================\n");
    printf(" EXPERIMENT 3: C4 Covariance Splitting & Decay Exponent\n");
    printf(" X_MAX=%llu  H=%d  threads=%d\n",
           (unsigned long long)X_MAX, H, omp_get_max_threads());
    printf("==========================================================\n\n");

    /* ---- 1. Primes ---- */
    printf("Step 1: Prime sieve...\n");
    int np;
    uint32_t *primes = prime_list(X_MAX + H + 64, &np);
    printf("  %d primes found.\n\n", np);

    /* ---- 2. Liouville sieve ---- */
    printf("Step 2: Liouville sieve...\n");
    uint8_t *lam = build_lam(X_MAX, np, primes);
    free(primes);

    /* Sanity */
    /* lambda(n) = 1 - 2*lam[n].
     * lambda(2)=-1 [Omega=1], lambda(4)=+1 [Omega=2], lambda(12)=-1 [Omega=3] */
    printf("  lambda(2)=%d lambda(4)=%d lambda(12)=%d (expect -1,+1,-1)\n\n",
           1-2*(int)lam[2], 1-2*(int)lam[4], 1-2*(int)lam[12]);

    /* ---- 3. For each k, compute statistics ---- */
    printf("Step 3: Computing C4, mu1, mu2, Cov for each k...\n\n");

    for (int ki = 0; ki < (int)N_K_VALS; ki++) {
        int k = K_VALS[ki];
        printf("---- k = %d, h = %d ----\n", k, H);
        printf("%-12s  %-12s  %-11s  %-11s  %-12s  %-12s  %-8s\n",
               "X", "C4", "mu1", "mu2", "mu1*mu2", "|Cov|", "Cov/C4");
        printf("%-12s  %-12s  %-11s  %-11s  %-12s  %-12s  %-8s\n",
               "------------","------------","-----------",
               "-----------","------------","------------","--------");

        double C4s[N_CP], mu1s[N_CP], mu2s[N_CP], Covs[N_CP], S2s[N_CP];
        compute_stats(lam, k, C4s, mu1s, mu2s, Covs, S2s);

        for (int ci = 0; ci < (int)N_CP; ci++) {
            if (CP[ci] > X_MAX) break;
            double mm = mu1s[ci]*mu2s[ci];
            double rat = (fabs(C4s[ci]) > 1e-12)
                         ? fabs(Covs[ci])/fabs(C4s[ci]) : 0.0;
            printf("%-12llu  %-+12.8f  %-+11.7f  %-+11.7f  %-12.4e  %-12.8f  %-8.5f\n",
                   (unsigned long long)CP[ci],
                   C4s[ci], mu1s[ci], mu2s[ci], mm,
                   fabs(Covs[ci]), rat);
        }

        /* Fit decay exponents */
        double alpha_C4  = fit_alpha(C4s,  CP, N_CP);
        double alpha_Cov = fit_alpha(Covs, CP, N_CP);
        double alpha_mu1 = fit_alpha(mu1s, CP, N_CP);
        printf("\n  Fitted: |C4|  ~ (log X)^{-%.3f}\n", alpha_C4);
        printf("  Fitted: |Cov| ~ (log X)^{-%.3f}\n", alpha_Cov);
        printf("  Fitted: |mu1| ~ (log X)^{-%.3f}\n\n", alpha_mu1);
        printf("  Goldbach requires alpha >= 2. Current: %.3f\n\n", alpha_C4);
    }

    /* ---- 4. Two-point decay: S2(X) = sum_{n<=X} lambda(n)*lambda(n+k) / X ---- */
    printf("---- Two-point partial sums ----\n");
    printf("  S2(X) = (1/X) sum_{n<=X} lambda(n)*lambda(n+k)\n\n");

    for (int ki = 0; ki < (int)N_K_VALS; ki++) {
        int kk = K_VALS[ki];
        printf("k = %d:\n", kk);
        printf("  %-12s  %-14s  %-12s\n", "X", "|S2|", "alpha_raw");
        double prev_S2 = 0.0; uint64_t prev_X = 0;
        for (int ci = 0; ci < (int)N_CP; ci++) {
            uint64_t Xc = CP[ci];
            if (Xc > X_MAX) break;
            int64_t s=0;
            #pragma omp parallel for reduction(+:s) schedule(static)
            for (uint64_t n=1; n<=Xc; n++) {
                uint8_t a = lam[n] ^ lam[n+kk];
                s += (int)(1 - 2*(int)a);
            }
            double S2 = fabs((double)s/(double)Xc);
            double alpha = 0.0;
            if (prev_X > 0 && prev_S2 > 1e-12 && S2 > 1e-12)
                alpha = -log(S2/prev_S2) /
                        log(log((double)Xc)/log((double)prev_X));
            printf("  %-12llu  %-14.9f  %-12.4f\n",
                   (unsigned long long)Xc, S2, alpha);
            prev_S2 = S2; prev_X = Xc;
        }
        printf("\n");
    }

    /* ---- 5. Summary table ---- */
    printf("==========================================================\n");
    printf(" SUMMARY: Fitted decay exponents alpha (|f| ~ (logX)^{-alpha})\n");
    printf("%-8s  %-10s  %-10s  %-10s  %-10s\n",
           "k", "alpha_C4", "alpha_Cov", "alpha_mu1", "alpha_S2");
    printf("%-8s  %-10s  %-10s  %-10s  %-10s\n",
           "--------","----------","----------","----------","----------");

    for (int ki = 0; ki < (int)N_K_VALS; ki++) {
        int k = K_VALS[ki];
        double C4s[N_CP], mu1s[N_CP], mu2s[N_CP], Covs[N_CP], S2s[N_CP];
        compute_stats(lam, k, C4s, mu1s, mu2s, Covs, S2s);
        double aC4  = fit_alpha(C4s,  CP, N_CP);
        double aCov = fit_alpha(Covs, CP, N_CP);
        double amu1 = fit_alpha(mu1s, CP, N_CP);
        double aS2  = fit_alpha(S2s,  CP, N_CP);
        printf("%-8d  %-10.3f  %-10.3f  %-10.3f  %-10.3f\n",
               k, aC4, aCov, amu1, aS2);
    }
    printf("\nGoldbach threshold: alpha >= 2\n");
    printf("Total elapsed: %.1fs\n", elapsed_s(t_start));

    free(lam);
    return 0;
}
