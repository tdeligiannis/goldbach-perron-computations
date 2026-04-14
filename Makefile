# Makefile for Goldbach computational experiments
# ================================================
#
# Paper: T. Kolokolnikov, "A reduction of binary Goldbach to a
#        four-point Chowla bound via the polynomial squaring identity"
#        (2026), Appendix A.
#
# Hardware: Pop!_OS Linux, 64-core CPU, 540 GB RAM
#
# Usage:
#   make all             # compile all C programs
#   make test            # quick smoke tests (~2 min)
#   make run-exp1        # full Experiment 1 (C4 covariance, ~8h)
#   make run-exp2        # full Experiment 2 (TK interactions, ~4h)
#   make run-exp4        # full Experiment 4 (wavelet, ~5 min at N=10B)
#   make test-python     # test all Python experiments (~30s)

CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
LDFLAGS = -lm -fopenmp

# C targets
C_TARGETS = four_point/c4_covariance \
            turan_kubilius/tk_two_prime \
            wavelet/goldbach_wavelet

.PHONY: all clean test test-python run-exp1 run-exp2 run-exp3 run-exp4 run-exp5

all: $(C_TARGETS)

four_point/c4_covariance: four_point/c4_covariance.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

turan_kubilius/tk_two_prime: turan_kubilius/tk_two_prime.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

wavelet/goldbach_wavelet: wavelet/goldbach_wavelet.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

# ── Quick tests (small X, ~2 min total) ──

test: test-c test-python

test-c: four_point/c4_covariance.c turan_kubilius/tk_two_prime.c wavelet/goldbach_wavelet.c
	@echo "=== C test: Experiment 1 (X=50M) ==="
	$(CC) $(CFLAGS) -DX_MAX=50000000ULL -o /tmp/test_c4 four_point/c4_covariance.c $(LDFLAGS)
	OMP_NUM_THREADS=4 /tmp/test_c4
	@echo ""
	@echo "=== C test: Experiment 2 (X=50M) ==="
	$(CC) $(CFLAGS) -DX_MAX=50000000ULL -o /tmp/test_tk turan_kubilius/tk_two_prime.c $(LDFLAGS)
	OMP_NUM_THREADS=4 /tmp/test_tk
	@echo ""
	@echo "=== C test: Experiment 4 (N=1B) ==="
	$(CC) $(CFLAGS) -o /tmp/test_wav wavelet/goldbach_wavelet.c $(LDFLAGS)
	/tmp/test_wav 1

test-python:
	@echo "=== Python test: Experiment 4 (wavelet, N=2^20) ==="
	cd wavelet && python3 wavelet_verification.py --test
	@echo ""
	@echo "=== Python test: Experiment 5c (expsums, X=10^5) ==="
	cd exponential_sums && python3 multiscale_expsums.py --test

# ── Full runs (production scale) ──

run-exp1: four_point/c4_covariance
	@echo "Running Experiment 1: C4 covariance splitting (X=10^10)..."
	OMP_NUM_THREADS=64 four_point/c4_covariance 2>&1 | tee results_exp1.txt

run-exp2: turan_kubilius/tk_two_prime
	@echo "Running Experiment 2: TK interactions (X=2×10^9)..."
	OMP_NUM_THREADS=64 turan_kubilius/tk_two_prime 2>&1 | tee results_exp2.txt

run-exp3:
	@echo "Running Experiment 3: Perron regularity (M=10^10)..."
	cd perron && python3 goldbach_perron_computation.py \
	  --mmax 10000000000 --cores 64 --outdir results 2>&1 | tee ../results_exp3.txt

run-exp4: wavelet/goldbach_wavelet
	@echo "Running Experiment 4: Wavelet verification (N=10B)..."
	cd wavelet && ./goldbach_wavelet 10 2>&1 | tee ../results_exp4.txt

run-exp5:
	@echo "Running Experiment 5c: Multi-scale expsums (X=10^8)..."
	cd exponential_sums && python3 multiscale_expsums.py --xmax 8 2>&1 | tee ../results_exp5c.txt

clean:
	rm -f $(C_TARGETS) /tmp/test_c4 /tmp/test_tk /tmp/test_wav
	rm -f results_exp*.txt
