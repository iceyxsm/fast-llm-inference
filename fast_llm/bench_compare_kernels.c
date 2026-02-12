/*
 * Kernel Comparison Benchmark
 * Compares all matmul implementations against the fast 6x16 baseline
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

/* Wrapper for asm_style kernel */
void wrapper_asm_style(const float* A, const dequantized_tensor_t* B, float* C,
                       int M, int N, int K) {
    matmul_dequantized_asm_style(A, B, C, M, N, K);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║         KERNEL COMPARISON - Against 6x16 ASM Baseline            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    
    /* Setup data */
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    dequantized_tensor_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    B.scales = (float*)aligned_malloc(m * sizeof(float), 32);
    B.rows = m;
    B.cols = n;
    B.original_bits = 8;
    
    randn(input, n);
    for (int i = 0; i < m * n; i++) B.weights[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < m; i++) B.scales[i] = 0.01f;
    
    /* Benchmark baseline (6x16 ASM) */
    printf("Testing: 6x16 ASM-Style (BASELINE)\n");
    
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    
    /* Benchmark */
    int iterations = 100;
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    double baseline_time = (get_time() - start) / iterations;
    double baseline_tok = 1.0 / (baseline_time * NUM_LAYERS * 2);
    
    printf("  Time: %.3f ms | Tok/sec: %.1f\n\n", baseline_time * 1000, baseline_tok);
    
    /* Test other kernels */
    typedef struct {
        const char* name;
        void (*func)(const float*, const dequantized_tensor_t*, float*, int, int, int);
        double time_ms;
        double tok_sec;
        double vs_baseline;
    } kernel_test_t;
    
    kernel_test_t tests[] = {
        {"Prefetch Optimized", matmul_dequantized_prefetch_optimized, 0, 0, 0},
        {"VNNI INT8", matmul_int8_vnni, 0, 0, 0},
        {"Ultra", matmul_int8_ultra, 0, 0, 0},
        {"Super", matmul_int8_super, 0, 0, 0},
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int t = 0; t < num_tests; t++) {
        printf("Testing: %s\n", tests[t].name);
        
        /* Warmup */
        for (int i = 0; i < 10; i++) {
            tests[t].func(input, &B, output, 1, m, n);
        }
        
        /* Benchmark */
        start = get_time();
        for (int i = 0; i < iterations; i++) {
            tests[t].func(input, &B, output, 1, m, n);
        }
        double time = (get_time() - start) / iterations;
        
        tests[t].time_ms = time * 1000;
        tests[t].tok_sec = 1.0 / (time * NUM_LAYERS * 2);
        tests[t].vs_baseline = baseline_time / time;
        
        printf("  Time: %.3f ms | Tok/sec: %.1f | vs Baseline: %.2fx\n\n", 
               tests[t].time_ms, tests[t].tok_sec, tests[t].vs_baseline);
    }
    
    /* Summary table */
    printf("╔════════════════════════╦═══════════╦═══════════╦════════════╗\n");
    printf("║ Kernel                 ║ Time (ms) ║ Tok/sec   ║ vs 6x16    ║\n");
    printf("╠════════════════════════╬═══════════╬═══════════╬════════════╣\n");
    printf("║ %-22s ║ %9.3f ║ %9.1f ║ %8s   ║\n", "6x16 ASM (baseline)", baseline_time * 1000, baseline_tok, "1.00x");
    
    for (int t = 0; t < num_tests; t++) {
        const char* status = tests[t].vs_baseline > 1.0 ? "✓" : "✗";
        printf("║ %-22s ║ %9.3f ║ %9.1f ║ %7.2fx %s ║\n", 
               tests[t].name, tests[t].time_ms, tests[t].tok_sec, 
               tests[t].vs_baseline, status);
    }
    printf("╚════════════════════════╩═══════════╩═══════════╩════════════╝\n");
    
    /* Find best */
    double best_time = baseline_time;
    const char* best_name = "6x16 ASM";
    
    for (int t = 0; t < num_tests; t++) {
        if (tests[t].time_ms / 1000.0 < best_time) {
            best_time = tests[t].time_ms / 1000.0;
            best_name = tests[t].name;
        }
    }
    
    printf("\n✓ BEST KERNEL: %s (%.1f tok/sec)\n", best_name, 1.0 / (best_time * NUM_LAYERS * 2));
    
    /* Cleanup */
    aligned_free(input);
    aligned_free(output);
    aligned_free(B.weights);
    aligned_free(B.scales);
    
    return 0;
}
