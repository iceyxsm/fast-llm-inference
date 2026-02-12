/*
 * Final Assembly-Style Benchmark
 * Tests all optimized kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* Declare all kernels */
extern void matmul_dequantized(const float* A, const dequantized_tensor_t* B,
                                float* C, int M, int N, int K);
extern void matmul_dequantized_best(const float* A, const dequantized_tensor_t* B,
                                     float* C, int M, int N, int K);
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
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
    for (int w = 0; w < 10; w++) {
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
    
    /* Calculate estimated full model speed */
    double ms_per_token = ms_per * 32 * 2;  /* 32 layers, 2 matmuls */
    double tok_per_sec = 1000.0 / ms_per_token;
    
    printf("%-25s: %6.3f ms | %5.1f GFLOPS | %5.2f tok/sec\n", 
           name, ms_per, gflops, tok_per_sec);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  FINAL ASSEMBLY-STYLE BENCHMARK\n");
    printf("  All Optimized Kernels Compared\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 200;
    
    /* Create weights */
    dequantized_tensor_t W_up, W_down;
    
    /* Fused gate+up: [2*inter, hidden] */
    W_up.rows = 2 * intermediate;
    W_up.cols = hidden;
    W_up.weights = aligned_malloc(2 * intermediate * hidden, 64);
    W_up.scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    /* Down: [hidden, intermediate] */
    W_down.rows = hidden;
    W_down.cols = intermediate;
    W_down.weights = aligned_malloc(hidden * intermediate, 64);
    W_down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < hidden; r++) {
        W_down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            W_down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Matrix Dimensions:\n");
    printf("  Gate+Up: [1, %d] @ [%d, %d] = [%d]\n", hidden, 2*intermediate, hidden, 2*intermediate);
    printf("  Down:    [1, %d] @ [%d, %d] = [%d]\n\n", intermediate, hidden, intermediate, hidden);
    
    printf("Benchmarking %d iterations each:\n\n", iterations);
    printf("Kernel                    |  Time   |  GFLOPS  | Tok/sec\n");
    printf("--------------------------|---------|----------|----------\n");
    
    /* Benchmark gate+up projection */
    benchmark_kernel("Gate+Up: Original", matmul_dequantized,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    
    benchmark_kernel("Gate+Up: Best", matmul_dequantized_best,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    
    benchmark_kernel("Gate+Up: ASM-Style", matmul_dequantized_asm_style,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    
    printf("\n");
    
    /* Benchmark down projection */
    benchmark_kernel("Down: Original", matmul_dequantized,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    
    benchmark_kernel("Down: Best", matmul_dequantized_best,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    
    benchmark_kernel("Down: ASM-Style", matmul_dequantized_asm_style,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    
    /* Find best and run full model test */
    printf("\n========================================\n");
    printf("  FULL MODEL TEST (Best Kernel)\n");
    printf("========================================\n\n");
    
    /* Quick test to determine best */
    double best_time = 1e9;
    void (*best_kernel)(const float*, const dequantized_tensor_t*, float*, int, int, int) = matmul_dequantized;
    const char* best_name = "Original";
    
    /* Test original */
    double start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized(input, &W_up, output_up, 1, 2*intermediate, hidden);
        matmul_dequantized(output_up, &W_down, output_down, 1, hidden, intermediate);
    }
    double t = (get_time_ms() - start) / 50.0;
    if (t < best_time) { best_time = t; best_kernel = matmul_dequantized; best_name = "Original"; }
    
    /* Test best */
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_best(input, &W_up, output_up, 1, 2*intermediate, hidden);
        matmul_dequantized_best(output_up, &W_down, output_down, 1, hidden, intermediate);
    }
    t = (get_time_ms() - start) / 50.0;
    if (t < best_time) { best_time = t; best_kernel = matmul_dequantized_best; best_name = "Best"; }
    
    /* Test asm-style */
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_asm_style(input, &W_up, output_up, 1, 2*intermediate, hidden);
        matmul_dequantized_asm_style(output_up, &W_down, output_down, 1, hidden, intermediate);
    }
    t = (get_time_ms() - start) / 50.0;
    if (t < best_time) { best_time = t; best_kernel = matmul_dequantized_asm_style; best_name = "ASM-Style"; }
    
    printf("Best kernel: %s\n", best_name);
    printf("Time per layer pair: %.3f ms\n", best_time);
    
    /* Full model benchmark */
    int num_tokens = 100;
    printf("\nRunning %d tokens with 32 layers...\n", num_tokens);
    
    start = get_time_ms();
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < 32; layer++) {
            /* Gate + Up */
            best_kernel(input, &W_up, output_up, 1, 2*intermediate, hidden);
            
            /* SwiGLU */
            for (int j = 0; j < intermediate; j++) {
                float g = output_up[j];
                float u = output_up[j + intermediate];
                float sig = 1.0f / (1.0f + expf(-g));
                output_up[j] = g * sig * u;
            }
            
            /* Down */
            best_kernel(output_up, &W_down, output_down, 1, hidden, intermediate);
            
            /* Residual */
            for (int j = 0; j < hidden; j++) {
                output_down[j] += input[j];
            }
            
            float* t = input; input = output_down; output_down = t;
        }
        
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    double elapsed = get_time_ms() - start;
    double tok_per_sec = num_tokens / (elapsed / 1000.0);
    
    printf("\n\n=== FINAL RESULTS ===\n");
    printf("Time: %.2f ms\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("TARGET: 25 tok/sec\n");
    printf("GAP: %.1f%% of target\n", (tok_per_sec / 25.0) * 100.0);
    printf("\n");
    
    if (tok_per_sec >= 25.0) {
        printf("🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉\n");
    } else if (tok_per_sec >= 20.0) {
        printf("⚠️ Close! (80%+)\n");
    } else {
        printf("Need more optimization\n");
    }
    printf("\n");
    
    /* Cleanup */
    aligned_free(W_up.weights);
    aligned_free(W_up.scales);
    aligned_free(W_down.weights);
    aligned_free(W_down.scales);
    aligned_free(input);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return 0;
}
