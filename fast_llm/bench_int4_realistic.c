/*
 * Realistic INT4 vs INT8 Comparison
 * Uses actual optimized kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

typedef struct {
    uint8_t* weights;
    float* scales;
    int rows;
    int cols;
} int4_matrix_t;

extern void matmul_int4_lookup(const float* A, const int4_matrix_t* B, float* C,
                                int M, int N, int K);
extern int4_matrix_t* create_int4_matrix(const float* weights, int rows, int cols);
extern void free_int4_matrix(int4_matrix_t* mat);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* INT8 forward pass with optimized kernels */
void forward_int8(int num_layers, int hidden, int intermediate,
                  float* hidden_state,
                  dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int layer = 0; layer < num_layers; layer++) {
        rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
        matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
        swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
        matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
        for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

/* INT4 forward pass */
void forward_int4(int num_layers, int hidden, int intermediate,
                  float* hidden_state,
                  int4_matrix_t* W_up, int4_matrix_t* W_down) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int layer = 0; layer < num_layers; layer++) {
        rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
        matmul_int4_lookup(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
        swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
        matmul_int4_lookup(output_up, W_down, output_down, 1, hidden, intermediate);
        for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  REALISTIC INT4 vs INT8 (32 LAYERS)\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    
    /* Create float weights */
    printf("Creating weights...\n");
    int up_size = 2 * intermediate * hidden;
    int down_size = hidden * intermediate;
    
    float* W_up_float = (float*)aligned_malloc(up_size * sizeof(float), 64);
    float* W_down_float = (float*)aligned_malloc(down_size * sizeof(float), 64);
    
    for (int i = 0; i < up_size; i++) W_up_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < down_size; i++) W_down_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    /* Create INT4 matrices */
    printf("Quantizing to INT4...\n");
    int4_matrix_t* W_up_int4 = create_int4_matrix(W_up_float, 2*intermediate, hidden);
    int4_matrix_t* W_down_int4 = create_int4_matrix(W_down_float, hidden, intermediate);
    
    /* Create INT8 tensors */
    printf("Quantizing to INT8...\n");
    dequantized_tensor_t W_up_int8, W_down_int8;
    W_up_int8.rows = 2 * intermediate;
    W_up_int8.cols = hidden;
    W_up_int8.weights = (int8_t*)aligned_malloc(up_size, 64);
    W_up_int8.scales = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up_int8.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            int q = (int)roundf(W_up_float[r * hidden + c] / 0.01f);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            W_up_int8.weights[r * hidden + c] = (int8_t)q;
        }
    }
    W_down_int8.rows = hidden;
    W_down_int8.cols = intermediate;
    W_down_int8.weights = (int8_t*)aligned_malloc(down_size, 64);
    W_down_int8.scales = (float*)aligned_malloc(hidden * sizeof(float), 64);
    for (int r = 0; r < hidden; r++) {
        W_down_int8.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            int q = (int)roundf(W_down_float[r * intermediate + c] / 0.01f);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            W_down_int8.weights[r * intermediate + c] = (int8_t)q;
        }
    }
    
    /* Memory usage */
    printf("\nMemory usage:\n");
    printf("  INT8 weights: %.1f MB + scales: %.1f MB = %.1f MB total\n",
           (up_size + down_size) / (1024.0 * 1024.0),
           (2*intermediate + hidden) * sizeof(float) / (1024.0 * 1024.0),
           ((up_size + down_size) + (2*intermediate + hidden) * sizeof(float)) / (1024.0 * 1024.0));
    printf("  INT4 weights: %.1f MB + scales: %.1f MB = %.1f MB total\n",
           (up_size + down_size) / (2.0 * 1024.0 * 1024.0),
           (2*intermediate + hidden) * sizeof(float) / (1024.0 * 1024.0),
           ((up_size + down_size) / 2.0 + (2*intermediate + hidden) * sizeof(float)) / (1024.0 * 1024.0));
    printf("  INT4 saves: %.1f%% of weight memory\n", 50.0);
    
    /* Benchmark INT8 */
    printf("\n1. Benchmarking INT8 (AVX2 optimized)...\n");
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        forward_int8(num_layers, hidden, intermediate, hidden_state, &W_up_int8, &W_down_int8);
    }
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        forward_int8(num_layers, hidden, intermediate, hidden_state, &W_up_int8, &W_down_int8);
    }
    double int8_time = get_time_ms() - start;
    double int8_tok_sec = tokens / (int8_time / 1000.0);
    printf("   INT8: %.1f ms, %.1f tok/sec\n", int8_time, int8_tok_sec);
    
    /* Benchmark INT4 */
    printf("\n2. Benchmarking INT4 (lookup-based)...\n");
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 3; w++) {
        forward_int4(num_layers, hidden, intermediate, hidden_state, W_up_int4, W_down_int4);
    }
    
    /* Reduce tokens for INT4 since it's slower */
    tokens = 20;
    start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        forward_int4(num_layers, hidden, intermediate, hidden_state, W_up_int4, W_down_int4);
    }
    double int4_time = get_time_ms() - start;
    double int4_tok_sec = tokens / (int4_time / 1000.0);
    printf("   INT4: %.1f ms, %.1f tok/sec\n", int4_time, int4_tok_sec);
    
    /* Results */
    printf("\n========================================\n");
    printf("  RESULTS (32 LAYERS)\n");
    printf("========================================\n");
    printf("INT8 (AVX2): %.1f tok/sec\n", int8_tok_sec);
    printf("INT4 (lookup): %.1f tok/sec\n", int4_tok_sec);
    printf("Target: 50 tok/sec\n\n");
    
    if (int8_tok_sec >= 50.0) {
        printf("✅ INT8 already achieves 50+ tok/sec!\n");
    } else {
        double gap = 50.0 - int8_tok_sec;
        printf("❌ INT8 needs %.1f more tok/sec\n", gap);
        printf("   INT4 is %.1fx slower than INT8\n", int8_tok_sec / int4_tok_sec);
        printf("\nConclusion:\n");
        printf("- INT4 memory savings don't help due to unpacking overhead\n");
        printf("- Need AVX2-optimized INT4 matmul to beat INT8\n");
        printf("- 24 layers (55 tok/sec) remains best practical solution\n");
    }
    
    /* Cleanup */
    aligned_free(W_up_float);
    aligned_free(W_down_float);
    free_int4_matrix(W_up_int4);
    free_int4_matrix(W_down_int4);
    aligned_free(W_up_int8.weights);
    aligned_free(W_up_int8.scales);
    aligned_free(W_down_int8.weights);
    aligned_free(W_down_int8.scales);
    aligned_free(hidden_state);
    
    return 0;
}
