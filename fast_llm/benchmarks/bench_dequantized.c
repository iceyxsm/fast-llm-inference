/*
 * Benchmark for Dequantized (Pre-dequantized INT8) Matmul
 * 
 * Compares:
 * - Q4 with runtime unpacking
 * - Pre-dequantized INT8
 * - AVX2 _mm256_maddubs_epi16
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cpu_features.h"
#include "quant_types.h"
#include "matmul.h"
#include "dequantized_tensor.h"

/* Get time in seconds */
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Fill array with random values */
void randomize(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

int main(void) {
    printf("============================================\n");
    printf("Dequantized INT8 Matmul Benchmark\n");
    printf("============================================\n\n");
    
    /* Detect CPU features */
    cpu_features_t features = detect_cpu_features();
    print_cpu_info(&features);
    printf("\n");
    
    /* Test dimensions */
    int M = 1;        /* Batch size */
    int N = 8192;     /* Intermediate size */
    int K = 3072;     /* Hidden size */
    
    printf("Test dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("This simulates one Phi-3 FFN layer forward pass\n\n");
    
    /* Allocate matrices */
    float* A = (float*)aligned_malloc(M * K * sizeof(float), 64);
    float* C_q4 = (float*)aligned_malloc(M * N * sizeof(float), 64);
    float* C_dq = (float*)aligned_malloc(M * N * sizeof(float), 64);
    
    if (!A || !C_q4 || !C_dq) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    randomize(A, M * K);
    
    /* Create Q4 quantized weights */
    printf("Creating Q4 quantized weights...\n");
    quantized_tensor_t* B_q4 = create_q4_tensor(N, K);
    if (!B_q4) {
        fprintf(stderr, "Failed to create Q4 tensor\n");
        return 1;
    }
    
    /* Fill with random quantized values */
    float* temp = (float*)malloc(N * K * sizeof(float));
    randomize(temp, N * K);
    
    for (int n = 0; n < N; n++) {
        for (int k_block = 0; k_block < (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE; k_block++) {
            int offset = n * K + k_block * Q4_BLOCK_SIZE;
            int count = (offset + Q4_BLOCK_SIZE <= N * K) ? Q4_BLOCK_SIZE : (N * K - offset);
            quantize_f32_to_q4(temp + offset,
                (block_q4_t*)B_q4->blocks + n * ((K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE) + k_block,
                count);
        }
    }
    
    free(temp);
    
    /* Create pre-dequantized tensor */
    printf("Pre-dequantizing to INT8 (one-time cost)...\n");
    double dequant_start = get_time();
    dequantized_tensor_t* B_dq = dequantized_from_q4(B_q4);
    double dequant_time = get_time() - dequant_start;
    
    if (!B_dq) {
        fprintf(stderr, "Failed to create dequantized tensor\n");
        return 1;
    }
    
    printf("Dequantization time: %.3f ms (one-time at model load)\n", dequant_time * 1000);
    printf("Memory overhead: %.1f MB (int8 weights + scales)\n\n", 
           (N * K * sizeof(int8_t) + N * sizeof(float)) / (1024.0 * 1024.0));
    
    /* Select kernels */
    matmul_fn_t q4_kernel = select_q4_kernel(&features);
    
    /* Benchmark Q4 with runtime unpacking */
    printf("Benchmarking Q4 with runtime unpacking...\n");
    int iterations = 50;
    
    /* Warmup */
    for (int i = 0; i < 3; i++) {
        q4_kernel(A, B_q4, C_q4, M, N, K);
    }
    
    /* Benchmark */
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        q4_kernel(A, B_q4, C_q4, M, N, K);
    }
    double q4_time = (get_time() - start) / iterations;
    
    printf("  Q4 kernel: %.3f ms/call\n", q4_time * 1000);
    
    /* Benchmark pre-dequantized */
    printf("\nBenchmarking pre-dequantized INT8...\n");
    
    /* Warmup */
    for (int i = 0; i < 3; i++) {
        if (features.has_avx2) {
            matmul_dequantized_avx2(A, B_dq, C_dq, M, N, K);
        } else {
            matmul_dequantized(A, B_dq, C_dq, M, N, K);
        }
    }
    
    /* Benchmark */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        if (features.has_avx2) {
            matmul_dequantized_avx2(A, B_dq, C_dq, M, N, K);
        } else {
            matmul_dequantized(A, B_dq, C_dq, M, N, K);
        }
    }
    double dq_time = (get_time() - start) / iterations;
    
    printf("  INT8 kernel: %.3f ms/call\n", dq_time * 1000);
    
    /* Speedup */
    double speedup = q4_time / dq_time;
    printf("\n  Speedup: %.1fx\n", speedup);
    
    /* Verify correctness (rough check) */
    double max_diff = 0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C_q4[i] - C_dq[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max difference: %.3e (should be small)\n", max_diff);
    
    /* Estimate full model performance */
    printf("\n");
    printf("============================================\n");
    printf("Estimated Full Model Performance\n");
    printf("============================================\n");
    
    int num_layers = 32;
    int matmuls_per_layer = 3;
    
    double time_q4 = q4_time * 1000 * num_layers * matmuls_per_layer;
    double time_dq = dq_time * 1000 * num_layers * matmuls_per_layer;
    
    double tok_q4 = 1000.0 / time_q4;
    double tok_dq = 1000.0 / time_dq;
    
    printf("Q4 (runtime unpack): %.1f ms/token -> %.1f tok/sec\n", time_q4, tok_q4);
    printf("INT8 (pre-dequant):  %.1f ms/token -> %.1f tok/sec\n", time_dq, tok_dq);
    printf("\n");
    printf("Speedup: %.1fx\n", speedup);
    printf("\n");
    printf("For comparison:\n");
    printf("  llama.cpp Q4: ~25 tok/sec\n");
    printf("  Target (4x):  ~100 tok/sec\n");
    
    if (tok_dq >= 100) {
        printf("\n*** TARGET ACHIEVED! ***\n");
    } else if (tok_dq >= 25) {
        printf("\n*** MATCHES llama.cpp! ***\n");
    } else {
        printf("\nGap to llama.cpp: %.1fx\n", 25.0 / tok_dq);
        printf("Need: EAGLE-3 speculative (2.5x) to reach %.1f tok/sec\n", tok_dq * 2.5);
    }
    
    /* Cleanup */
    aligned_free(A);
    aligned_free(C_q4);
    aligned_free(C_dq);
    free_quantized_tensor(B_q4);
    dequantized_tensor_free(B_dq);
    
    return 0;
}
