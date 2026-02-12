/*
 * Final 50 tok/sec Benchmark
 * 
 * Compares all optimization levels:
 * 1. Reference (scalar)
 * 2. AVX2 basic
 * 3. VNNI INT8
 * 4. Ultra-optimized
 * 5. Super-optimized (32 rows)
 * 
 * Target: 50+ tok/sec on DDR4-3200
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

/* Benchmark a single matmul implementation */
typedef void (*matmul_func_t)(const float*, const dequantized_tensor_t*, float*, int, int, int);

typedef struct {
    const char* name;
    matmul_func_t func;
    double time_ms;
    double bandwidth_gb_s;
    double tok_per_sec;
    double speedup;
} benchmark_result_t;

void benchmark_matmul_impl(matmul_func_t func, const char* name, 
                           const float* A, const dequantized_tensor_t* B, float* C,
                           int M, int N, int K, benchmark_result_t* result) {
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        func(A, B, C, M, N, K);
    }
    
    /* Benchmark */
    int iterations = 100;
    double start = get_time();
    
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, M, N, K);
    }
    
    double end = get_time();
    double avg_time = (end - start) / iterations;
    
    /* Calculate metrics */
    double bytes = (double)N * K * sizeof(int8_t) + N * sizeof(float) + 
                   K * sizeof(float) + N * sizeof(float);
    double bandwidth = bytes / avg_time / 1e9;
    double tok_per_sec = 1.0 / (avg_time * NUM_LAYERS * 2);
    
    result->name = name;
    result->time_ms = avg_time * 1000;
    result->bandwidth_gb_s = bandwidth;
    result->tok_per_sec = tok_per_sec;
    result->speedup = 1.0; /* Will be set relative to baseline */
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║         FINAL 50 TOK/SEC OPTIMIZATION BENCHMARK                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Hardware Target: DDR4-3200 (61 GB/s measured)                   ║\n");
    printf("║ Model: Phi-3-mini (32 layers, 3072 hidden, 8192 intermediate)   ║\n");
    printf("║ CPU: AVX2 with OpenMP                                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
#ifdef _OPENMP
    printf("OpenMP Threads: %d\n\n", omp_get_max_threads());
#endif
    
    srand(42);
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    
    /* Allocate data */
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    /* Create INT8 weights */
    dequantized_tensor_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    B.scales = (float*)aligned_malloc(m * sizeof(float), 32);
    B.rows = m;
    B.cols = n;
    B.original_bits = 8;
    
    randn(input, n);
    for (int i = 0; i < m * n; i++) B.weights[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < m; i++) B.scales[i] = 0.01f;
    
    /* Benchmark all implementations */
    benchmark_result_t results[4];
    
    printf("Running benchmarks...\n\n");
    
    /* Reference */
    printf("1. Reference (matmul_dequantized_asm_style)...\n");
    benchmark_matmul_impl(matmul_dequantized_asm_style, "Reference", 
                          input, &B, output, 1, m, n, &results[0]);
    
    /* VNNI */
    printf("2. VNNI INT8 (matmul_int8_vnni)...\n");
    benchmark_matmul_impl(matmul_int8_vnni, "VNNI INT8", 
                          input, &B, output, 1, m, n, &results[1]);
    
    /* Ultra */
    printf("3. Ultra-optimized (matmul_int8_ultra)...\n");
    benchmark_matmul_impl(matmul_int8_ultra, "Ultra", 
                          input, &B, output, 1, m, n, &results[2]);
    
    /* Super */
    printf("4. Super-optimized (matmul_int8_super)...\n");
    benchmark_matmul_impl(matmul_int8_super, "Super", 
                          input, &B, output, 1, m, n, &results[3]);
    
    /* Calculate speedups */
    for (int i = 1; i < 4; i++) {
        results[i].speedup = results[0].time_ms / results[i].time_ms;
    }
    
    /* Print results table */
    printf("\n");
    printf("╔════════════════════╦═══════════╦════════════╦════════════╦══════════╗\n");
    printf("║ Implementation     ║ Time (ms) ║ Bandwidth  ║ Tok/sec    ║ Speedup  ║\n");
    printf("╠════════════════════╬═══════════╬════════════╬════════════╬══════════╣\n");
    
    for (int i = 0; i < 4; i++) {
        printf("║ %-18s ║ %9.3f ║ %6.1f GB/s ║ %6.1f     ║ %6.2fx ║\n",
               results[i].name,
               results[i].time_ms,
               results[i].bandwidth_gb_s,
               results[i].tok_per_sec,
               results[i].speedup);
    }
    
    printf("╚════════════════════╩═══════════╩════════════╩════════════╩══════════╝\n");
    
    /* Analysis */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                        ANALYSIS                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    
    double best_tok = results[3].tok_per_sec;
    double target = 50.0;
    
    printf("║ Best Performance:  %.1f tok/sec                                 ║\n", best_tok);
    printf("║ Target:            %.0f tok/sec                                   ║\n", target);
    printf("║ Gap to Target:     %+.1f tok/sec (%+.0f%%)                        ║\n",
           best_tok - target, (best_tok / target - 1.0) * 100);
    
    if (best_tok >= target) {
        printf("║ Status:            ✅ TARGET ACHIEVED!                           ║\n");
    } else {
        printf("║ Status:            ⚠️  BELOW TARGET                              ║\n");
        double needed_speedup = target / best_tok;
        printf("║ Need:              %.2fx more speedup to reach target           ║\n", needed_speedup);
    }
    
    /* Memory bandwidth analysis */
    double measured_bw = results[3].bandwidth_gb_s;
    double system_bw = 61.0;
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Memory Bandwidth:                                                ║\n");
    printf("║   System:          %.1f GB/s (DDR4-3200)                        ║\n", system_bw);
    printf("║   Achieved:        %.1f GB/s (%.0f%% utilization)                  ║\n",
           measured_bw, measured_bw / system_bw * 100);
    
    if (measured_bw / system_bw < 0.8) {
        printf("║   Status:          ⚠️  Not saturating memory bandwidth           ║\n");
        printf("║   Potential:       Up to %.0f tok/sec if bandwidth saturated    ║\n",
               best_tok * system_bw / measured_bw);
    } else {
        printf("║   Status:          ✅ Memory bandwidth well utilized             ║\n");
    }
    
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    /* Cleanup */
    aligned_free(input);
    aligned_free(output);
    aligned_free(B.weights);
    aligned_free(B.scales);
    
    return 0;
}
