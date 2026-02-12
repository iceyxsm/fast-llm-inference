/*
 * Matrix Multiplication Benchmark
 * Tests Q2, Q4, Q8 kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cpu_features.h"
#include "quant_types.h"
#include "matmul.h"

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

/* Benchmark matmul kernel */
void benchmark_kernel(const char* name, matmul_fn_t kernel,
                      const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K, int iterations) {
    
    /* Warmup */
    for (int i = 0; i < 3; i++) {
        kernel(A, B, C, M, N, K);
    }
    
    /* Benchmark */
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        kernel(A, B, C, M, N, K);
    }
    double elapsed = get_time() - start;
    
    /* Calculate metrics */
    double time_per_call = elapsed / iterations * 1000.0; /* ms */
    double flops = 2.0 * M * N * K; /* multiply-adds */
    double gflops = flops / (time_per_call / 1000.0) / 1e9;
    
    /* Memory bandwidth */
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = (size_t)B->n_blocks * ((B->bits == 2) ? sizeof(block_q2_t) :
                                             (B->bits == 4) ? sizeof(block_q4_t) :
                                             sizeof(block_q8_t));
    size_t bytes_C = M * N * sizeof(float);
    size_t total_bytes = bytes_A + bytes_B + bytes_C;
    double bandwidth = (double)total_bytes / (time_per_call / 1000.0) / 1e9; /* GB/s */
    
    printf("%s:\n", name);
    printf("  Time: %.3f ms/call\n", time_per_call);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    printf("  Memory: %.2f GB/s\n", bandwidth);
}

int main(void) {
    printf("============================================\n");
    printf("Fast LLM - Matrix Multiplication Benchmark\n");
    printf("============================================\n\n");
    
    /* Detect CPU features */
    cpu_features_t features = detect_cpu_features();
    print_cpu_info(&features);
    printf("\n");
    
    /* Test dimensions (Phi-3 FFN layer) */
    int M = 1;        /* Batch size */
    int N = 8192;     /* Intermediate size */
    int K = 3072;     /* Hidden size */
    
    printf("Test dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("This simulates one Phi-3 FFN layer forward pass\n\n");
    
    /* Allocate matrices */
    float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * N * sizeof(float));
    
    if (!A || !C) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    randomize(A, M * K);
    
    /* Create quantized weights */
    printf("Creating quantized weights...\n");
    quantized_tensor_t* B_q2 = create_q2_tensor(N, K);
    quantized_tensor_t* B_q4 = create_q4_tensor(N, K);
    quantized_tensor_t* B_q8 = create_q8_tensor(N, K);
    
    if (!B_q2 || !B_q4 || !B_q8) {
        fprintf(stderr, "Failed to create quantized tensors\n");
        return 1;
    }
    
    /* Fill with random quantized values */
    float* temp = (float*)malloc(N * K * sizeof(float));
    randomize(temp, N * K);
    
    for (int n = 0; n < N; n++) {
        for (int k_block = 0; k_block < (K + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE; k_block++) {
            int offset = n * K + k_block * Q2_BLOCK_SIZE;
            int count = (offset + Q2_BLOCK_SIZE <= N * K) ? Q2_BLOCK_SIZE : (N * K - offset);
            quantize_f32_to_q2(temp + offset, 
                (block_q2_t*)B_q2->blocks + n * ((K + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE) + k_block, 
                count);
        }
    }
    
    for (int n = 0; n < N; n++) {
        for (int k_block = 0; k_block < (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE; k_block++) {
            int offset = n * K + k_block * Q4_BLOCK_SIZE;
            int count = (offset + Q4_BLOCK_SIZE <= N * K) ? Q4_BLOCK_SIZE : (N * K - offset);
            quantize_f32_to_q4(temp + offset,
                (block_q4_t*)B_q4->blocks + n * ((K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE) + k_block,
                count);
        }
    }
    
    for (int n = 0; n < N; n++) {
        for (int k_block = 0; k_block < (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE; k_block++) {
            int offset = n * K + k_block * Q8_BLOCK_SIZE;
            int count = (offset + Q8_BLOCK_SIZE <= N * K) ? Q8_BLOCK_SIZE : (N * K - offset);
            quantize_f32_to_q8(temp + offset,
                (block_q8_t*)B_q8->blocks + n * ((K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE) + k_block,
                count);
        }
    }
    
    free(temp);
    
    printf("Done.\n\n");
    
    /* Run benchmarks */
    int iterations = 100;
    
    printf("Running benchmarks (%d iterations each)...\n\n", iterations);
    
    /* Q2 benchmark */
    if (features.has_avx512f) {
        benchmark_kernel("Q2 AVX-512", q2_matmul_avx512, A, B_q2, C, M, N, K, iterations);
    } else {
        benchmark_kernel("Q2 Scalar", q2_matmul, A, B_q2, C, M, N, K, iterations);
    }
    printf("\n");
    
    /* Q4 benchmark */
    if (features.has_avx512f) {
        benchmark_kernel("Q4 AVX-512", q4_matmul_avx512, A, B_q4, C, M, N, K, iterations);
    } else {
        benchmark_kernel("Q4 Scalar", q4_matmul, A, B_q4, C, M, N, K, iterations);
    }
    printf("\n");
    
    /* Q8 benchmark */
    if (features.has_avx512f) {
        benchmark_kernel("Q8 AVX-512", q8_matmul_avx512, A, B_q8, C, M, N, K, iterations);
    } else {
        benchmark_kernel("Q8 Scalar", q8_matmul, A, B_q8, C, M, N, K, iterations);
    }
    
    /* Estimate full model performance */
    printf("\n");
    printf("============================================\n");
    printf("Estimated Full Model Performance\n");
    printf("============================================\n");
    
    /* Phi-3 has 32 layers, each with ~3 matmuls */
    int num_layers = 32;
    int matmuls_per_layer = 3;
    
    /* Use Q2 time for estimate (fastest) */
    double q2_time_ms = 0;
    if (features.has_avx512f) {
        double start = get_time();
        for (int i = 0; i < 10; i++) q2_matmul_avx512(A, B_q2, C, M, N, K);
        q2_time_ms = (get_time() - start) / 10 * 1000;
    } else {
        double start = get_time();
        for (int i = 0; i < 10; i++) q2_matmul(A, B_q2, C, M, N, K);
        q2_time_ms = (get_time() - start) / 10 * 1000;
    }
    
    double time_per_token_ms = q2_time_ms * num_layers * matmuls_per_layer;
    double tokens_per_sec = 1000.0 / time_per_token_ms;
    
    printf("Estimated tokens/sec (Q2, 32 layers): %.1f tok/sec\n", tokens_per_sec);
    printf("\n");
    printf("For comparison:\n");
    printf("  llama.cpp Q4 (similar CPU): ~25 tok/sec\n");
    printf("  Target (4x faster): ~100 tok/sec\n");
    
    if (tokens_per_sec >= 100) {
        printf("\n*** TARGET ACHIEVED! ***\n");
    } else {
        printf("\nGap to target: %.1fx\n", 100 / tokens_per_sec);
        printf("Need: EAGLE-3 speculative decoding\n");
    }
    
    /* Cleanup */
    free(A);
    free(C);
    free_quantized_tensor(B_q2);
    free_quantized_tensor(B_q4);
    free_quantized_tensor(B_q8);
    
    return 0;
}
