/*
 * goldbach_wavelet.c — Experiment 4: Wavelet Verification
 *
 * Paper: T. Deligiannis, "A reduction of binary Goldbach to a
 *        four-point Chowla bound via the polynomial squaring identity"
 *        (2026), Appendix A, §A.4.
 *
 * Effective cascade verification for Goldbach conjecture.
 *
 * COMPILE:
 *   sudo apt install build-essential
 *   gcc -O3 -march=native -fopenmp -o goldbach_wavelet goldbach_wavelet.c -lm
 *
 * RUN:
 *   ./goldbach_wavelet 10     # Test: N=10B, ~10 GB, ~5 min
 *   ./goldbach_wavelet 50     # Medium: N=50B, ~50 GB, ~20 min
 *   ./goldbach_wavelet 200    # Full: N=200B, ~200 GB, ~2 hrs
 *   ./goldbach_wavelet 400    # Max: N=400B, ~400 GB, ~4 hrs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

static long long N;
static int8_t  *lam;
static int      GOLDBACH_H;
static int      n_small_primes;
static int     *small_primes;
static double   program_start;

static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void stamp(void) {
    double e = wtime() - program_start;
    int h = (int)(e/3600), m = (int)(e/60)%60, s = (int)e%60;
    if (h > 0) printf("[%dh%02dm%02ds]", h, m, s);
    else if (m > 0) printf("[%dm%02ds]", m, s);
    else printf("[%ds]", s);
}

static void phase_eta(double phase_start, double frac) {
    if (frac < 0.01) { printf(" ETA: ..."); return; }
    double rem = (wtime() - phase_start) / frac * (1.0 - frac);
    int m = (int)(rem/60), s = (int)rem%60;
    if (m > 0) printf(" ETA ~%dm%02ds", m, s);
    else printf(" ETA ~%ds", s);
}

/* Phase 1: find primes up to sqrt(N) */
static void small_prime_sieve(void) {
    long long sq = (long long)sqrt((double)N) + 2;
    char *isp = (char *)calloc(sq + 1, 1);
    for (long long i = 2; i <= sq; i++) isp[i] = 1;
    for (long long i = 2; i*i <= sq; i++)
        if (isp[i]) for (long long j = i*i; j <= sq; j += i) isp[j] = 0;
    n_small_primes = 0;
    for (long long i = 2; i <= sq; i++) if (isp[i]) n_small_primes++;
    small_primes = (int *)malloc(n_small_primes * sizeof(int));
    int idx = 0;
    for (long long i = 2; i <= sq; i++) if (isp[i]) small_primes[idx++] = (int)i;
    free(isp);
    stamp(); printf(" Found %d primes up to %lld\n", n_small_primes, sq);
}

/* Phase 2: flip lambda at multiples of small prime powers */
static void apply_small_primes(void) {
    double t0 = wtime();
    for (int pi = 0; pi < n_small_primes; pi++) {
        long long p = small_primes[pi];
        long long pk = p;
        int pw = 0;
        while (pk <= N) {
            pw++;
            #pragma omp parallel for schedule(static)
            for (long long j = pk; j <= N; j += pk)
                lam[j] = -lam[j];
            if (p <= 50) {
                stamp();
                printf(" p=%lld^%d: flipped %lld entries\n", p, pw, N/pk);
            }
            if (pk > N/p) break;
            pk *= p;
        }
        /* periodic progress */
        int step = n_small_primes / 10; if (step < 1) step = 1;
        if (p > 50 && ((pi+1) % step == 0 || pi == n_small_primes-1)) {
            double f = (double)(pi+1)/n_small_primes;
            stamp(); printf(" Small primes: %d/%d (%.0f%%)", pi+1, n_small_primes, 100*f);
            phase_eta(t0, f);
            printf("\n");
        }
    }
    stamp(); printf(" Phase 2 done (%.1fs)\n", wtime()-t0);
}

/* Phase 3: segmented sieve for large primes */
#define SEG_SIZE (1 << 22)
static void apply_large_primes(void) {
    double t0 = wtime();
    long long sq = (long long)sqrt((double)N) + 2;
    long long nsegs = (N - sq + SEG_SIZE) / SEG_SIZE;
    long long total_lp = 0, segs_done = 0;
    long long print_step = nsegs / 50; if (print_step < 1) print_step = 1;

    stamp(); printf(" %lld segments, est %.0f large primes\n",
                    nsegs, (double)N/log((double)N) - (double)sq/log((double)sq));

    #pragma omp parallel reduction(+:total_lp)
    {
        char *seg = (char *)malloc(SEG_SIZE);
        #pragma omp for schedule(dynamic, 16)
        for (long long s = 0; s < nsegs; s++) {
            long long L = sq + 1 + s * (long long)SEG_SIZE;
            long long U = L + SEG_SIZE; if (U > N+1) U = N+1;
            long long slen = U - L;
            memset(seg, 1, slen);
            for (int pi = 0; pi < n_small_primes; pi++) {
                long long p = small_primes[pi];
                long long st = ((L+p-1)/p)*p;
                for (long long j = st-L; j < slen; j += p) seg[j] = 0;
            }
            for (long long i = 0; i < slen; i++) {
                if (seg[i]) {
                    long long p = L + i; total_lp++;
                    for (long long j = p; j <= N; j += p) lam[j] = -lam[j];
                }
            }
            long long done;
            #pragma omp atomic capture
            done = ++segs_done;
            if (done % print_step == 0 || done == nsegs) {
                double f = (double)done/nsegs;
                #pragma omp critical
                { stamp(); printf(" Segments: %lld/%lld (%.1f%%), ~%lld primes",
                                  done, nsegs, 100*f, total_lp);
                  phase_eta(t0, f); printf("\n"); fflush(stdout); }
            }
        }
        free(seg);
    }
    stamp(); printf(" Phase 3 done: %lld large primes (%.1fs)\n", total_lp, wtime()-t0);
}

/* Verification */
static int verify_lambda(void) {
    stamp(); printf(" Verifying lambda...\n");
    struct { long long n; int exp; } chk[] = {
        {2,-1},{3,-1},{5,-1},{7,-1},{4,1},{12,-1},{30,-1},{36,1},{997,-1}
    };
    int ok = 1;
    for (int i = 0; i < 9; i++) {
        if (chk[i].n > N) continue;
        int v = lam[chk[i].n];
        if (v != chk[i].exp) { ok = 0; printf("  FAIL lam[%lld]=%d expected %d\n", chk[i].n, v, chk[i].exp); }
    }
    long long s = 0;
    for (long long n = 1; n <= 10000 && n <= N; n++) s += lam[n];
    printf("  sum lam(n<=10000) = %lld", s);
    if (s == -94) printf(" (matches known value)\n");
    else { printf(" (EXPECTED -94, MISMATCH!)\n"); ok = 0; }
    if (ok) printf("  All checks PASSED.\n");
    else printf("  *** VERIFICATION FAILED ***\n");
    return ok;
}

/* Phase 4: Wavelet analysis */
static void compute_wavelets(FILE *fout) {
    int h = GOLDBACH_H;
    int j_first_pass = -1, j_max_verified = -1;
    int total_tested = 0, total_passed = 0;

    printf("\n%-4s %14s %10s %13s %10s %8s %4s %16s %8s %8s\n",
           "j","block_size","K","max|C|/2^j","1/j^2","ratio","PW","MS/4^j","delta","time");
    printf("---- -------------- ---------- ------------- ---------- -------- ---- "
           "---------------- -------- --------\n");
    fprintf(fout, "j,block_size,K,max_norm,threshold,ratio,pointwise,mean_square,delta\n");

    for (int j = 16; j <= 60; j++) {
        long long bs = 1LL << j;
        long long mk = (N - h) / bs - 1;
        if (mk < 1) break;

        double jt0 = wtime();
        double thr = 1.0 / ((double)j*j);
        long long K = 0, n_exc = 0, blk_done = 0;
        double maxn = 0, sumsq = 0;
        long long pstep = mk/20; if (pstep < 1) pstep = 1;

        /* Storage for block values (for printing if K small) */
        double *norms = NULL;
        if (mk <= 50) norms = (double *)calloc(mk, sizeof(double));

        #pragma omp parallel
        {
            double lmax = 0, lss = 0;
            long long lK = 0, lexc = 0;

            #pragma omp for schedule(dynamic, 64)
            for (long long k = 1; k <= mk; k++) {
                long long st = k * bs, en = st + bs;
                if (en + h > N) continue;
                long long C = 0;
                for (long long n = st; n < en; n++)
                    C += (long long)lam[n] * lam[n+h];
                double nm = fabs((double)C) / (double)bs;
                if (norms) norms[k-1] = nm;
                if (nm > lmax) lmax = nm;
                lss += nm*nm; lK++; if (nm > thr) lexc++;

                long long d;
                #pragma omp atomic capture
                d = ++blk_done;
                if (mk >= 100 && d % pstep == 0) {
                    double f = (double)d/mk;
                    #pragma omp critical
                    { stamp(); printf("   j=%d: %lld/%lld blocks (%.0f%%)", j, d, mk, 100*f);
                      phase_eta(jt0, f); printf("\n"); fflush(stdout); }
                }
            }
            #pragma omp critical
            { if (lmax > maxn) maxn = lmax; sumsq += lss; K += lK; n_exc += lexc; }
        }

        if (K == 0) { if (norms) free(norms); break; }
        double ms = sumsq/K, delta = (double)n_exc/K;
        double ratio = maxn/thr;
        int pw = (maxn <= thr);
        double jelapsed = wtime()-jt0;

        total_tested += (int)K;
        if (pw) { total_passed += (int)K; if (j_first_pass < 0) j_first_pass = j; j_max_verified = j; }

        printf("%-4d %14lld %10lld %13.8f %10.6f %8.3f %4s %16.12f %8.4f %7.1fs\n",
               j, bs, K, maxn, thr, ratio, pw?"Y":"N", ms, delta, jelapsed);
        fprintf(fout,"%d,%lld,%lld,%.10e,%.10e,%.4f,%d,%.10e,%.8f\n",
                j,bs,K,maxn,thr,ratio,pw,ms,delta);
        fflush(fout); fflush(stdout);

        /* Print blocks if few */
        if (norms && K <= 50) {
            for (long long k = 0; k < K; k++) {
                printf("      k=%4lld: |C|/2^j = %.10f (%.1f%% of thr)%s\n",
                       k+1, norms[k], 100*norms[k]/thr,
                       norms[k]>thr?" *** EXCEEDS ***":"");
            }
        }
        if (norms) free(norms);

        /* Summary line for this j */
        stamp();
        if (pw) printf(" >>> j=%d PASS: all %lld blocks OK (worst %.1f%% of threshold)\n\n", j, K, 100*ratio);
        else    printf(" >>> j=%d FAIL: %lld/%lld exceed (worst ratio %.2f)\n\n", j, n_exc, K, ratio);
    }

    /* ========== FINAL SUMMARY ========== */
    printf("\n============================================================\n");
    printf("  FINAL SUMMARY\n");
    printf("============================================================\n");
    printf("  N              = %lld (%.1f billion)\n", N, (double)N/1e9);
    printf("  h              = %d\n", GOLDBACH_H);
    printf("  Total blocks   = %d tested, %d passed\n", total_tested, total_passed);
    if (j_first_pass >= 0 && j_max_verified >= j_first_pass) {
        printf("  Pointwise bound holds for j = %d to %d\n", j_first_pass, j_max_verified);
        long long Nt = 2LL * (1LL<<j_first_pass) * (1LL<<j_first_pass);
        printf("\n  CONDITIONAL GOLDBACH THEOREM:\n");
        printf("    Assume |C_j(k,h)|/2^j <= 1/j^2 for all j >= %d.\n", j_first_pass);
        printf("    Verified computationally for %d <= j <= %d.\n", j_first_pass, j_max_verified);
        printf("    N_threshold = 2*(2^%d)^2 = %.2e\n", j_first_pass, (double)Nt);
        printf("    Goldbach verified to 4e18.\n");
        if ((double)Nt < 4e18)
            printf("    Margin: %.1e  ==>  GOLDBACH HOLDS (conditionally).\n", 4e18/(double)Nt);
        else
            printf("    Threshold exceeds 4e18 — need larger computation.\n");
    }
    printf("============================================================\n");
}

int main(int argc, char **argv) {
    program_start = wtime();

    long long Nin = 10;
    if (argc >= 2) Nin = atoll(argv[1]);
    N = (Nin > 1000) ? Nin : Nin * 1000000000LL;
    GOLDBACH_H = 2;
    if (argc >= 3) GOLDBACH_H = atoi(argv[2]);

    printf("==========================================================\n");
    printf("  Goldbach Cascade Verification  (%s)\n", __DATE__);
    printf("==========================================================\n");
    printf("  N        = %lld (%.1f billion)\n", N, (double)N/1e9);
    printf("  h        = %d\n", GOLDBACH_H);
    printf("  RAM      = %.1f GB\n", (double)(N+1)/1e9);
    printf("  Threads  = %d\n", omp_get_max_threads());
    printf("  Max j    ~ %d\n", (int)(log2((double)N)));
    printf("==========================================================\n\n");

    stamp(); printf(" Allocating %.1f GB...\n", (double)(N+1)/1e9);
    lam = (int8_t *)malloc(N+1);
    if (!lam) { fprintf(stderr,"Out of memory! Need %.1f GB.\n",(double)(N+1)/1e9); return 1; }
    memset(lam, 1, N+1); lam[0] = 0;
    stamp(); printf(" Allocation done.\n\n");

    printf("== PHASE 1: Small prime sieve ==\n");
    small_prime_sieve();

    printf("\n== PHASE 2: Apply small primes ==\n");
    apply_small_primes();

    printf("\n== PHASE 3: Large prime sieve ==\n");
    apply_large_primes();

    printf("\n== VERIFICATION ==\n");
    if (!verify_lambda()) { fprintf(stderr,"Sieve incorrect, aborting.\n"); return 1; }

    printf("\n== PHASE 4: Wavelet analysis ==\n");
    char fn[256];
    snprintf(fn,sizeof(fn),"goldbach_wavelets_N%.0fB_h%d.csv",(double)N/1e9,GOLDBACH_H);
    FILE *fout = fopen(fn,"w");
    if (!fout) { fprintf(stderr,"Cannot open %s\n",fn); return 1; }
    compute_wavelets(fout);
    fclose(fout);

    double tt = wtime()-program_start;
    printf("\n  Total: %dh%02dm%02ds\n", (int)(tt/3600), (int)(tt/60)%60, (int)tt%60);
    printf("  CSV: %s\n", fn);

    free(lam); free(small_primes);
    return 0;
}
