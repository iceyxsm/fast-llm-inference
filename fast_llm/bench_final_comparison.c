/*
 * Final Comparison - All Kernels
 * Determines the best approach
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "dequantized_tensor.h"
#include "cpu_features.h"

extern void matmul_dequantized(const float* A, const dequantized_tensor_t* B,
                                float* C, int M, int N, int K);
extern void matmul_dequantized_best(const float* A, const dequantized_tensor_t* B,
                                     float* C, int M, int N, int K);
extern void matmul_dequantized_llamafile(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

void benchmark_kernel(const char* name,
                      void (*kernel)(const float*, const dequantized_tensor_t*, float*, int, int, int),
                      const float* A, const dequantized_tensor_t* B, float* C,
                      int M, int N, int K, int iterations) {
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        kernel(A, B, C, M, N, K);
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        kernel(A, B, C, M, N, K);
    }
    double elapsed = get_time_ms() - start;
    
    double ms_per = elapsed / iterations;
    double gflops = (2.0 * M * N * K * iterations) / (elapsed * 1e6);
    
    printf("  %-20s: %6.3f ms, %5.2f GFLOPS\n", name, ms_per, gflops);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  FINAL KERNEL COMPARISON\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 100;
    
    /* Create weights */
    dequantized_tensor_t W;
    W.rows = intermediate;
    W.cols = hidden;
    W.weights = aligned_malloc(intermediate * hidden, 64);
    W.scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    for (int r = 0; r < intermediate; r++) {
        W.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* output = aligned_malloc(intermediate * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Matrix: [1, %d] @ [%d, %d] = [%d]\n\n", hidden, intermediate, hidden, intermediate);
    printf("Benchmarking %d iterations each:\n\n", iterations);
    
    /* Benchmark all kernels */
    benchmark_kernel("Original", matmul_dequantized,
                     input, &W, output, 1, intermediate, hidden, iterations);
    
    benchmark_kernel("Best (4x unroll)", matmul_dequantized_best,
                     input, &W, output, 1, intermediate, hidden, iterations);
    
    benchmark_kernel("Llamafile (8x)", matmul_dequantized_llamafile,
                     input, &W, output, 1, intermediate, hidden, iterations);
    
    /* Estimate full model speed */
    printf("\n=== ESTIMATED FULL MODEL SPEED ===\n");
    printf("(32 layers, 2 matmuls per layer)\n\n");
    
    double best_ms = 0;  /* Will be filled from benchmark */
    
    /* Get actual timing for best kernel */
    double start = get_time_ms();
    for (int i = 0; i < 100; i++) {
        matmul_dequantized_best(input, &W, output, 1, intermediate, hidden);
    }
    best_ms = (get_time_ms() - start) / 100.0;
    
    double ms_per_token = best_ms * 32 * 2;  /* 32 layers, 2 matmuls */
    double tok_per_sec = 1000.0 / ms_per_token;
    
    printf("Best kernel time: %.3f ms/matmul\n", best_ms);
    printf("Estimated speed: %.2f tok/sec\n", tok_per_sec);
    printf("vs llama.cpp: %.1f%%\n", (tok_per_sec / 25.0) * 100.0);
    printf("\n");
    
    if (tok_per_sec >= 25.0) {
        printf("✅ TARGET ACHIEVED!\n");
    } else {
        printf("⚠️  Need %.1fx more speed\n", 25.0 / tok_per_sec);
    }
    printf("\n");
    
    /* Cleanup */
    aligned_free(W.weights);
    aligned_free(W.scales);
    aligned_free(input);
    aligned_free(output);
    
    return 0;
}
