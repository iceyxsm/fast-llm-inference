/*
 * INT8 x INT8 Benchmark
 * Compares INT8 matmul (2x throughput) vs float matmul
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

extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);
extern void matmul_dequantized_int8xint8(const float* A, const dequantized_tensor_t* B,
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
    
    printf("%-25s: %6.3f ms | %5.1f GFLOPS\n", name, ms_per, gflops);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  INT8 x INT8 vs FLOAT Benchmark\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 100;
    
    /* Create INT8 weights */
    dequantized_tensor_t W_up, W_down;
    
    /* Gate+Up: [2*inter, hidden] = [16384, 3072] */
    W_up.rows = 2 * intermediate;
    W_up.cols = hidden;
    W_up.weights = aligned_malloc(2 * intermediate * hidden, 64);
    W_up.scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        /* Random int8 values */
        W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    /* Down: [hidden, intermediate] = [3072, 8192] */
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
    
    printf("Gate+Up (%d x %d):\n", 2*intermediate, hidden);
    benchmark_kernel("6x16 ASM-Style (float)", matmul_dequantized_asm_style,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    benchmark_kernel("INT8xINT8 (maddubs)", matmul_dequantized_int8xint8,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    
    printf("\nDown (%d x %d):\n", hidden, intermediate);
    benchmark_kernel("6x16 ASM-Style (float)", matmul_dequantized_asm_style,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    benchmark_kernel("INT8xINT8 (maddubs)", matmul_dequantized_int8xint8,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    
    /* Calculate potential tok/sec */
    printf("\n========================================\n");
    printf("  SPEEDUP ANALYSIS\n");
    printf("========================================\n\n");
    
    /* Run quick benchmark to get timing */
    double start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_asm_style(input, &W_up, output_up, 1, 2*intermediate, hidden);
    }
    double t_up_asm = (get_time_ms() - start) / 50.0;
    
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_int8xint8(input, &W_up, output_up, 1, 2*intermediate, hidden);
    }
    double t_up_int8 = (get_time_ms() - start) / 50.0;
    
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_asm_style(output_up, &W_down, output_down, 1, hidden, intermediate);
    }
    double t_down_asm = (get_time_ms() - start) / 50.0;
    
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_int8xint8(output_up, &W_down, output_down, 1, hidden, intermediate);
    }
    double t_down_int8 = (get_time_ms() - start) / 50.0;
    
    double t_total_asm = t_up_asm + t_down_asm;
    double t_total_int8 = t_up_int8 + t_down_int8;
    
    double tok_asm = 1000.0 / t_total_asm / 32.0;
    double tok_int8 = 1000.0 / t_total_int8 / 32.0;
    
    printf("Gate+Up: ASM=%.3fms, INT8=%.3fms (%.2fx)\n", 
           t_up_asm, t_up_int8, t_up_asm/t_up_int8);
    printf("Down:    ASM=%.3fms, INT8=%.3fms (%.2fx)\n", 
           t_down_asm, t_down_int8, t_down_asm/t_down_int8);
    printf("\nTotal per layer: ASM=%.3fms, INT8=%.3fms\n", t_total_asm, t_total_int8);
    printf("\nEstimated tok/sec:\n");
    printf("  ASM-Style: %.2f tok/sec\n", tok_asm);
    printf("  INT8xINT8: %.2f tok/sec\n", tok_int8);
    printf("\nTarget: 50 tok/sec\n");
    printf("INT8 Gap: %.1f%%\n", (tok_int8 / 50.0) * 100.0);
    
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
