/*
 * Approach 2: Q4_K 4-bit vs INT8 8-bit Benchmark
 * 
 * Q4_K uses 4.5 bits/weight vs 8 bits for INT8
 * Expected: ~1.8x speedup from 50% memory bandwidth reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "matmul_optimized.h"
#include "ggml_quants.h"
#include "dequantized_tensor.h"

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#define HIDDEN_SIZE 3072
#define INTERMEDIATE_SIZE 8192
#define NUM_LAYERS 32

static double get_time(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static void randn(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

/* Benchmark INT8 matmul (current best) */
double benchmark_int8(int iterations, int n, int m) {
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    dequantized_tensor_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    B.scales = (float*)aligned_malloc(m * sizeof(float), 32);
    B.rows = m;
    B.cols = n;
    
    randn(input, n);
    for (int i = 0; i < m * n; i++) B.weights[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < m; i++) B.scales[i] = 0.01f;
    
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    
    /* Benchmark */
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    double elapsed = get_time() - start;
    
    aligned_free(input);
    aligned_free(output);
    aligned_free(B.weights);
    aligned_free(B.scales);
    
    return elapsed / iterations;
}

/* Benchmark Q4_K matmul */
double benchmark_q4k(int iterations, int n, int m) {
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    int kblocks = n / 256;
    size_t num_blocks = (size_t)m * kblocks;
    block_q4_K* weights = (block_q4_K*)aligned_malloc(num_blocks * sizeof(block_q4_K), 32);
    
    randn(input, n);
    for (size_t i = 0; i < num_blocks; i++) {
        weights[i].scales[0] = (uint8_t)(rand() % 64);
        weights[i].scales[1] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) {
            weights[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        matmul_q4_K_avx2(n, m, output, weights, input);
    }
    
    /* Benchmark */
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        matmul_q4_K_avx2(n, m, output, weights, input);
    }
    double elapsed = get_time() - start;
    
    aligned_free(input);
    aligned_free(output);
    aligned_free(weights);
    
    return elapsed / iterations;
}

/* Full model simulation with Q4_K */
double benchmark_full_model_q4k(int num_layers) {
    int hidden = HIDDEN_SIZE;
    int intermediate = INTERMEDIATE_SIZE;
    int iterations = 20;
    
    /* Setup Q4_K weights for gate, up, down projections */
    int kblocks_hidden = hidden / 256;
    int kblocks_intermediate = intermediate / 256;
    
    size_t gate_blocks = (size_t)intermediate * kblocks_hidden;
    size_t down_blocks = (size_t)hidden * kblocks_intermediate;
    
    block_q4_K* w_gate = (block_q4_K*)aligned_malloc(gate_blocks * sizeof(block_q4_K), 32);
    block_q4_K* w_up = (block_q4_K*)aligned_malloc(gate_blocks * sizeof(block_q4_K), 32);
    block_q4_K* w_down = (block_q4_K*)aligned_malloc(down_blocks * sizeof(block_q4_K), 32);
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* gate_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    
    /* Initialize */
    randn(hidden_state, hidden);
    for (size_t i = 0; i < gate_blocks; i++) {
        w_gate[i].scales[0] = w_up[i].scales[0] = (uint8_t)(rand() % 64);
        w_gate[i].scales[1] = w_up[i].scales[1] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) {
            w_gate[i].qs[j] = w_up[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    for (size_t i = 0; i < down_blocks; i++) {
        w_down[i].scales[0] = (uint8_t)(rand() % 64);
        w_down[i].scales[1] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) {
            w_down[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    
    /* Warmup */
    for (int iter = 0; iter < 5; iter++) {
        for (int layer = 0; layer < num_layers; layer++) {
            matmul_q4_K_avx2(hidden, intermediate, gate_out, w_gate, hidden_state);
            matmul_q4_K_avx2(hidden, intermediate, up_out, w_up, hidden_state);
            for (int i = 0; i < intermediate; i++) {
                gate_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
            }
            matmul_q4_K_avx2(intermediate, hidden, hidden_state, w_down, gate_out);
        }
    }
    
    /* Benchmark */
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        for (int layer = 0; layer < num_layers; layer++) {
            matmul_q4_K_avx2(hidden, intermediate, gate_out, w_gate, hidden_state);
            matmul_q4_K_avx2(hidden, intermediate, up_out, w_up, hidden_state);
            for (int i = 0; i < intermediate; i++) {
                gate_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
            }
            matmul_q4_K_avx2(intermediate, hidden, hidden_state, w_down, gate_out);
        }
    }
    double elapsed = get_time() - start;
    
    aligned_free(w_gate);
    aligned_free(w_up);
    aligned_free(w_down);
    aligned_free(hidden_state);
    aligned_free(gate_out);
    aligned_free(up_out);
    
    return elapsed / iterations;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  APPROACH 2: Q4_K 4-bit vs INT8 8-bit Benchmark                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    int iterations = 100;
    
    printf("Matrix size: %d x %d\n", n, m);
    printf("Iterations: %d\n\n", iterations);
    
    /* Benchmark single matmul */
    printf("Single Matmul Comparison:\n");
    printf("  Testing INT8... ");
    fflush(stdout);
    double int8_time = benchmark_int8(iterations, n, m);
    double int8_tok = 1.0 / (int8_time * NUM_LAYERS * 2);
    printf("%.3f ms (%.1f tok/sec)\n", int8_time * 1000, int8_tok);
    
    printf("  Testing Q4_K... ");
    fflush(stdout);
    double q4k_time = benchmark_q4k(iterations, n, m);
    double q4k_tok = 1.0 / (q4k_time * NUM_LAYERS * 2);
    printf("%.3f ms (%.1f tok/sec)\n", q4k_time * 1000, q4k_tok);
    
    double speedup = int8_time / q4k_time;
    printf("\n  Speedup: %.2fx\n", speedup);
    
    /* Full model simulation */
    printf("\nFull Model Simulation (32 layers):\n");
    
    printf("  Testing Q4_K full model... ");
    fflush(stdout);
    double q4k_full = benchmark_full_model_q4k(32);
    double q4k_full_tok = 1.0 / q4k_full;
    printf("%.3f ms/token (%.1f tok/sec)\n", q4k_full * 1000, q4k_full_tok);
    
    /* Memory bandwidth analysis */
    printf("\nMemory Bandwidth Analysis:\n");
    
    /* INT8: 1 byte per weight */
    double int8_bytes = (double)m * n * 1.0 + m * sizeof(float) + n * sizeof(float);
    double int8_bw = int8_bytes / int8_time / 1e9;
    printf("  INT8: %.1f GB/s (%.0f%% of DDR4-3200)\n", int8_bw, int8_bw / 61.0 * 100);
    
    /* Q4_K: 4.5 bits = 0.5625 bytes per weight */
    double q4k_bytes = (double)m * n * 0.5625 + m * sizeof(float) + n * sizeof(float);
    double q4k_bw = q4k_bytes / q4k_time / 1e9;
    printf("  Q4_K: %.1f GB/s (%.0f%% of DDR4-3200)\n", q4k_bw, q4k_bw / 61.0 * 100);
    
    /* Summary */
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  INT8 8-bit:  %.1f tok/sec (baseline)                           ║\n", int8_tok);
    printf("║  Q4_K 4-bit:  %.1f tok/sec (%.2fx speedup)                      ║\n", q4k_tok, speedup);
    printf("║  Q4_K Full:   %.1f tok/sec (32 layers)                          ║\n", q4k_full_tok);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    
    if (q4k_full_tok >= 50.0) {
        printf("║  ✅ Q4_K alone achieves 50 tok/sec target!                       ║\n");
    } else {
        printf("║  ⚠️  Need additional optimizations                               ║\n");
        printf("║     Combine with --layers 20 for best results                   ║\n");
    }
    
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
