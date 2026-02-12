/*
 * Batched Inference Benchmark
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
extern void matmul_batched_2x(const float* A0, const float* A1,
                               const dequantized_tensor_t* B,
                               float* C0, float* C1,
                               int M, int N, int K);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  BATCHED INFERENCE BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    
    /* Create weights */
    dequantized_tensor_t W_up, W_down;
    
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
    
    /* Buffers for 2 tokens */
    float* input0 = aligned_malloc(hidden * sizeof(float), 64);
    float* input1 = aligned_malloc(hidden * sizeof(float), 64);
    float* out_up0 = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* out_up1 = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* out_down0 = aligned_malloc(hidden * sizeof(float), 64);
    float* out_down1 = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input0[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        input1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    int iterations = 50;
    
    /* Benchmark separate vs batched */
    printf("Gate+Up Projection (%d x %d):\n\n", 2*intermediate, hidden);
    
    /* Separate: 2 individual matmuls */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input0, &W_up, out_up0, 1, 2*intermediate, hidden);
        matmul_dequantized_asm_style(input1, &W_up, out_up1, 1, 2*intermediate, hidden);
    }
    double t_separate = (get_time_ms() - start) / iterations;
    printf("Separate (2x single): %.3f ms total\n", t_separate);
    
    /* Batched: 2 tokens at once */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_batched_2x(input0, input1, &W_up, out_up0, out_up1, 1, 2*intermediate, hidden);
    }
    double t_batched = (get_time_ms() - start) / iterations;
    printf("Batched (2x together): %.3f ms total\n", t_batched);
    
    printf("\nSpeedup: %.2fx\n", t_separate / t_batched);
    
    /* Down projection */
    printf("\nDown Projection (%d x %d):\n\n", hidden, intermediate);
    
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(out_up0, &W_down, out_down0, 1, hidden, intermediate);
        matmul_dequantized_asm_style(out_up1, &W_down, out_down1, 1, hidden, intermediate);
    }
    double t_down_sep = (get_time_ms() - start) / iterations;
    printf("Separate (2x single): %.3f ms total\n", t_down_sep);
    
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_batched_2x(out_up0, out_up1, &W_down, out_down0, out_down1, 1, hidden, intermediate);
    }
    double t_down_batch = (get_time_ms() - start) / iterations;
    printf("Batched (2x together): %.3f ms total\n", t_down_batch);
    
    printf("\nSpeedup: %.2fx\n", t_down_sep / t_down_batch);
    
    /* Full model estimate for batched inference */
    double t_total_batch = t_batched + t_down_batch;
    /* 2 tokens per batch, 32 layers */
    double tok_per_sec = 2000.0 / t_total_batch / 32.0;
    
    printf("\n=== BATCHED PERFORMANCE ===\n");
    printf("Per layer (2 tokens): %.3f ms\n", t_total_batch);
    printf("Throughput: %.2f tokens/sec\n", tok_per_sec);
    printf("Effective:  %.2f tok/sec (2x batch size)\n", tok_per_sec);
    printf("\nTarget: 50 tok/sec\n");
    printf("Gap: %.1f%%\n", (tok_per_sec / 50.0) * 100.0);
    
    /* Cleanup */
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(input0); aligned_free(input1);
    aligned_free(out_up0); aligned_free(out_up1);
    aligned_free(out_down0); aligned_free(out_down1);
    
    return 0;
}
