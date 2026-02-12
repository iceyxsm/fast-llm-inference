/*
 * Optimized Kernel Dispatch
 * 
 * Automatically selects the best kernel implementation based on:
 * - Hardware capabilities (AVX2, AVX-512)
 * - Matrix dimensions
 * - Memory pressure
 * - Quantization format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "matmul_optimized.h"
#include "cpu_features.h"

/* Global configuration */
static matmul_config_t g_config = {0};
static int g_initialized = 0;

#ifdef _WIN32
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

/* Initialize with defaults */
matmul_config_t matmul_get_default_config(void) {
    matmul_config_t config = {0};
    
    /* Enable all optimizations by default */
    config.use_prefetching = 1;
    config.prefetch_distance_l1 = 4;    /* 256 bytes */
    config.prefetch_distance_l2 = 16;   /* 1KB */
    config.prefetch_distance_l3 = 64;   /* 4KB */
    config.use_fused_ops = 1;
    config.use_q4_k = 1;
    
    /* Auto-detect thread count */
    const char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        config.num_threads = atoi(omp_threads);
    } else {
        config.num_threads = 0; /* Use all available */
    }
    
    return config;
}

void matmul_set_config(const matmul_config_t* config) {
    if (config) {
        g_config = *config;
    }
}

const matmul_config_t* matmul_get_config(void) {
    if (!g_initialized) {
        g_config = matmul_get_default_config();
        g_initialized = 1;
    }
    return &g_config;
}

/* Detect hardware capabilities and optimal settings */
static void detect_hardware(void) {
    /* Already detected in cpu_features.c */
}

/* 
 * Select best matmul implementation
 * Strategy:
 * - For Q4_K weights: use matmul_q4_K_optimized
 * - For INT8 weights with prefetching: use matmul_dequantized_prefetch_optimized
 * - For large matrices: use streaming variant
 * - Otherwise: use matmul_dequantized_asm_style
 */
void matmul_optimized_dispatch(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
) {
    const matmul_config_t* config = matmul_get_config();
    
    /* For single-token inference (M=1), we want optimal memory bandwidth usage */
    if (M == 1) {
        /* Check if we should use prefetching */
        if (config->use_prefetching && N >= 512) {
            /* Large N benefits from aggressive prefetching */
            matmul_dequantized_prefetch_optimized(A, B, C, M, N, K);
        } else if (N >= 4096) {
            /* Very large matrices use streaming prefetch */
            matmul_dequantized_streaming(A, B, C, M, N, K);
        } else {
            /* Standard optimized implementation */
            extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                                      float* C, int M, int N, int K);
            matmul_dequantized_asm_style(A, B, C, M, N, K);
        }
    } else {
        /* Batch processing - use standard optimized */
        extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                                  float* C, int M, int N, int K);
        matmul_dequantized_asm_style(A, B, C, M, N, K);
    }
}

/* Dispatch for Q4_K quantized matmul */
void matmul_q4_K_dispatch(
    int n, int m, float* s,
    const void* vx, const float* y
) {
    const matmul_config_t* config = matmul_get_config();
    
    if (config->use_q4_k) {
        /* Use optimized Q4_K implementation */
        matmul_q4_K_optimized(n, m, s, vx, y);
    } else {
        /* Fall back to reference implementation */
        extern void ggml_gemv_q4_K(int n, int m, float* s, const void* vx, const float* y);
        ggml_gemv_q4_K(n, m, s, vx, y);
    }
}

/* 
 * Performance profiling hooks
 * Track which kernels are being used for optimization decisions
 */
typedef struct {
    unsigned long long calls;
    unsigned long long cycles;
    const char* name;
} kernel_perf_t;

#define MAX_PERF_COUNTERS 16
static kernel_perf_t g_perf_counters[MAX_PERF_COUNTERS];
static int g_perf_num_counters = 0;

/* Register a performance counter */
int perf_register(const char* name) {
    if (g_perf_num_counters >= MAX_PERF_COUNTERS) return -1;
    int id = g_perf_num_counters++;
    g_perf_counters[id].name = name;
    g_perf_counters[id].calls = 0;
    g_perf_counters[id].cycles = 0;
    return id;
}

/* Start timing */
#if defined(_WIN32) && defined(__AVX2__)
#include <intrin.h>
static inline unsigned long long rdtsc(void) {
    return __rdtsc();
}
#else
static inline unsigned long long rdtsc(void) {
#ifdef __x86_64__
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
#else
    return 0;
#endif
}
#endif

void perf_start(int id) {
    if (id < 0 || id >= g_perf_num_counters) return;
    g_perf_counters[id].cycles -= rdtsc();
}

void perf_end(int id) {
    if (id < 0 || id >= g_perf_num_counters) return;
    g_perf_counters[id].cycles += rdtsc();
    g_perf_counters[id].calls++;
}

/* Print performance report */
void perf_report(void) {
    printf("\n=== Kernel Performance Report ===\n");
    for (int i = 0; i < g_perf_num_counters; i++) {
        kernel_perf_t* p = &g_perf_counters[i];
        if (p->calls > 0) {
            printf("%s: %llu calls, %llu cycles/call\n", 
                   p->name, p->calls, p->cycles / p->calls);
        }
    }
}

/* 
 * Auto-tuning for prefetch distances
 * Run microbenchmarks to find optimal prefetch distances for this hardware
 */
void matmul_autotune_prefetch(void) {
    printf("Auto-tuning prefetch distances...\n");
    
    /* Test configurations */
    int test_l1[] = {2, 4, 8, 16};
    int test_l2[] = {8, 16, 32, 64};
    int test_l3[] = {32, 64, 128, 256};
    
    double best_time = 1e9;
    int best_l1 = 4, best_l2 = 16, best_l3 = 64;
    
    /* Allocate test matrices */
    int N = 8192;
    int K = 3072;
    float* A = (float*)aligned_malloc(K * sizeof(float), 32);
    float* C = (float*)aligned_malloc(N * sizeof(float), 32);
    
    /* Create dummy quantized tensor */
    typedef struct {
        int8_t* weights;
        float* scales;
        int rows;
        int cols;
        int original_bits;
    } dummy_dequant_t;
    
    dummy_dequant_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)N * K, 32);
    B.scales = (float*)aligned_malloc(N * sizeof(float), 32);
    B.rows = N;
    B.cols = K;
    B.original_bits = 8;
    
    /* Initialize with random data */
    for (int i = 0; i < K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N * K; i++) B.weights[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < N; i++) {
        B.scales[i] = 0.01f;
    }
    
    /* Test each configuration */
    for (int i1 = 0; i1 < 4; i1++) {
        for (int i2 = 0; i2 < 4; i2++) {
            for (int i3 = 0; i3 < 4; i3++) {
                g_config.prefetch_distance_l1 = test_l1[i1];
                g_config.prefetch_distance_l2 = test_l2[i2];
                g_config.prefetch_distance_l3 = test_l3[i3];
                
                /* Warmup */
                for (int iter = 0; iter < 5; iter++) {
                    matmul_dequantized_prefetch_optimized(A, (dequantized_tensor_t*)&B, C, 1, N, K);
                }
                
                /* Time it */
                double start = omp_get_wtime();
                for (int iter = 0; iter < 20; iter++) {
                    matmul_dequantized_prefetch_optimized(A, (dequantized_tensor_t*)&B, C, 1, N, K);
                }
                double end = omp_get_wtime();
                double time = end - start;
                
                if (time < best_time) {
                    best_time = time;
                    best_l1 = test_l1[i1];
                    best_l2 = test_l2[i2];
                    best_l3 = test_l3[i3];
                }
            }
        }
    }
    
    /* Set best configuration */
    g_config.prefetch_distance_l1 = best_l1;
    g_config.prefetch_distance_l2 = best_l2;
    g_config.prefetch_distance_l3 = best_l3;
    
    printf("Optimal prefetch distances: L1=%d, L2=%d, L3=%d\n", best_l1, best_l2, best_l3);
    printf("Best time: %.3f ms\n", best_time * 1000 / 20);
    
    /* Cleanup */
    aligned_free(A);
    aligned_free(C);
    aligned_free(B.weights);
    aligned_free(B.scales);
}

/* External declaration for omp_get_wtime */
extern double omp_get_wtime(void);
