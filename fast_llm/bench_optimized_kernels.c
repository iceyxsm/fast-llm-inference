/*
 * Benchmark for Optimized Kernels
 * Tests AVX2-optimized Q4_K and VNNI INT8 implementations
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

#ifdef __AVX2__
#include <immintrin.h>
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

/* Benchmark Q4_K AVX2 kernel */
void benchmark_q4_k_avx2(void) {
    printf("\n=== Q4_K AVX2 Optimized ===\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    int kblocks = n / 256;
    
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
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
    int iterations = 100;
    double start = get_time();
    
    for (int i = 0; i < iterations; i++) {
        matmul_q4_K_avx2(n, m, output, weights, input);
    }
    
    double end = get_time();
    double avg_time = (end - start) / iterations;
    
    /* Q4_K: 4.5 bits per weight = 0.5625 bytes per weight */
    double bytes_per_matmul = (size_t)m * n * 0.5625 + n * sizeof(float) + m * sizeof(float);
    double bandwidth = bytes_per_matmul / avg_time / 1e9;
    double tokens_per_sec = 1.0 / (avg_time * NUM_LAYERS * 2);
    
    printf("  Time: %.3f ms\n", avg_time * 1000);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth);
    printf("  Tok/sec: %.1f\n", tokens_per_sec);
    
    aligned_free(input);
    aligned_free(output);
    aligned_free(weights);
}

/* Benchmark VNNI INT8 kernel */
void benchmark_int8_vnni(void) {
    printf("\n=== INT8 VNNI Optimized ===\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    /* Create dequantized tensor */
    dequantized_tensor_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    B.scales = (float*)aligned_malloc(m * sizeof(float), 32);
    B.rows = m;
    B.cols = n;
    B.original_bits = 8;
    
    randn(input, n);
    for (int i = 0; i < m * n; i++) B.weights[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < m; i++) B.scales[i] = 0.01f;
    
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        matmul_int8_vnni(input, &B, output, 1, m, n);
    }
    
    /* Benchmark */
    int iterations = 100;
    double start = get_time();
    
    for (int i = 0; i < iterations; i++) {
        matmul_int8_vnni(input, &B, output, 1, m, n);
    }
    
    double end = get_time();
    double avg_time = (end - start) / iterations;
    
    double bytes_per_matmul = (size_t)m * n * sizeof(int8_t) + n * sizeof(float) + m * sizeof(float);
    double bandwidth = bytes_per_matmul / avg_time / 1e9;
    double tokens_per_sec = 1.0 / (avg_time * NUM_LAYERS * 2);
    
    printf("  Time: %.3f ms\n", avg_time * 1000);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth);
    printf("  Tok/sec: %.1f\n", tokens_per_sec);
    
    aligned_free(input);
    aligned_free(output);
    aligned_free(B.weights);
    aligned_free(B.scales);
}

/* Compare with reference implementation */
void benchmark_comparison(void) {
    printf("\n=== Comparison: Optimized vs Reference ===\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    int kblocks = n / 256;
    
    /* Setup Q4_K weights */
    size_t num_blocks = (size_t)m * kblocks;
    block_q4_K* weights = (block_q4_K*)aligned_malloc(num_blocks * sizeof(block_q4_K), 32);
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output_ref = (float*)aligned_malloc(m * sizeof(float), 32);
    float* output_opt = (float*)aligned_malloc(m * sizeof(float), 32);
    
    randn(input, n);
    for (size_t i = 0; i < num_blocks; i++) {
        weights[i].scales[0] = (uint8_t)(rand() % 64);
        weights[i].scales[1] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) {
            weights[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    
    /* Time reference (ggml) */
    double start = get_time();
    for (int i = 0; i < 50; i++) {
        ggml_gemv_q4_K(n, m, output_ref, weights, input);
    }
    double time_ref = (get_time() - start) / 50;
    
    /* Time AVX2 optimized */
    start = get_time();
    for (int i = 0; i < 50; i++) {
        matmul_q4_K_avx2(n, m, output_opt, weights, input);
    }
    double time_avx2 = (get_time() - start) / 50;
    
    /* Verify correctness */
    float max_error = 0.0f;
    for (int i = 0; i < m; i++) {
        float err = fabsf(output_ref[i] - output_opt[i]);
        if (err > max_error) max_error = err;
    }
    
    printf("  Reference:  %.3f ms\n", time_ref * 1000);
    printf("  AVX2:       %.3f ms\n", time_avx2 * 1000);
    printf("  Speedup:    %.2fx\n", time_ref / time_avx2);
    printf("  Max error:  %e\n", max_error);
    
    aligned_free(weights);
    aligned_free(input);
    aligned_free(output_ref);
    aligned_free(output_opt);
}

/* Full model simulation */
void benchmark_full_model(void) {
    printf("\n=== Full Model Simulation ===\n");
    
    int hidden = HIDDEN_SIZE;
    int intermediate = INTERMEDIATE_SIZE;
    
    /* Setup FFN weights (Q4_K) */
    int kblocks = hidden / 256;
    size_t gate_blocks = (size_t)intermediate * kblocks;
    block_q4_K* w_gate = (block_q4_K*)aligned_malloc(gate_blocks * sizeof(block_q4_K), 32);
    block_q4_K* w_up = (block_q4_K*)aligned_malloc(gate_blocks * sizeof(block_q4_K), 32);
    
    /* Setup down projection (INT8) */
    dequantized_tensor_t w_down;
    w_down.weights = (int8_t*)aligned_malloc((size_t)hidden * intermediate, 32);
    w_down.scales = (float*)aligned_malloc(hidden * sizeof(float), 32);
    w_down.rows = hidden;
    w_down.cols = intermediate;
    
    /* Initialize */
    for (size_t i = 0; i < gate_blocks; i++) {
        w_gate[i].scales[0] = w_up[i].scales[0] = (uint8_t)(rand() % 64);
        w_gate[i].scales[1] = w_up[i].scales[1] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) {
            w_gate[i].qs[j] = w_up[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    for (int i = 0; i < hidden * intermediate; i++) {
        w_down.weights[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < hidden; i++) {
        w_down.scales[i] = 0.01f;
    }
    
    float* input = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* gate_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* swiglu_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* output = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    randn(input, hidden);
    
    /* Warmup */
    for (int i = 0; i < 5; i++) {
        matmul_q4_K_avx2(hidden, intermediate, gate_out, w_gate, input);
        matmul_q4_K_avx2(hidden, intermediate, up_out, w_up, input);
        for (int j = 0; j < intermediate; j++) {
            swiglu_out[j] = gate_out[j] * up_out[j]; /* Simplified SwiGLU */
        }
        matmul_int8_vnni(swiglu_out, &w_down, output, 1, hidden, intermediate);
    }
    
    /* Benchmark one layer */
    int iterations = 20;
    double start = get_time();
    
    for (int i = 0; i < iterations; i++) {
        /* FFN Gate */
        matmul_q4_K_avx2(hidden, intermediate, gate_out, w_gate, input);
        /* FFN Up */
        matmul_q4_K_avx2(hidden, intermediate, up_out, w_up, input);
        /* SwiGLU multiply */
        for (int j = 0; j < intermediate; j++) {
            swiglu_out[j] = gate_out[j] * up_out[j];
        }
        /* FFN Down */
        matmul_int8_vnni(swiglu_out, &w_down, output, 1, hidden, intermediate);
    }
    
    double end = get_time();
    double avg_time = (end - start) / iterations;
    
    double tokens_per_sec = 1.0 / (avg_time * NUM_LAYERS);
    
    printf("  One layer:     %.3f ms\n", avg_time * 1000);
    printf("  Full model:    %.3f ms\n", avg_time * 1000 * NUM_LAYERS);
    printf("  Tok/sec:       %.1f\n", tokens_per_sec);
    
    aligned_free(w_gate);
    aligned_free(w_up);
    aligned_free(w_down.weights);
    aligned_free(w_down.scales);
    aligned_free(input);
    aligned_free(gate_out);
    aligned_free(up_out);
    aligned_free(swiglu_out);
    aligned_free(output);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    printf("Optimized Kernel Benchmark\n");
    printf("==========================\n");
    
#ifdef _OPENMP
    printf("Threads: %d\n", omp_get_max_threads());
#endif
    
#ifdef __AVX2__
    printf("AVX2: Yes\n");
#else
    printf("AVX2: No\n");
#endif
    
    srand(42);
    
    benchmark_q4_k_avx2();
    benchmark_int8_vnni();
    benchmark_comparison();
    benchmark_full_model();
    
    printf("\nDone!\n");
    return 0;
}
