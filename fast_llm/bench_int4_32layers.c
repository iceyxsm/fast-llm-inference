/*
 * INT4 Quantization Benchmark for 32 Layers
 * Goal: 50+ tok/sec without layer reduction
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

/* External INT4 functions */
typedef struct int4_tensor_t int4_tensor_t;
extern int4_tensor_t* create_int4_tensor(const float* weights, int rows, int cols, int block_size);
extern void free_int4_tensor(int4_tensor_t* tensor);
extern void matmul_int4_lut(const float* A, const int4_tensor_t* B, float* C,
                             int M, int N, int K);

/* External kernels */
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Simulate INT4 forward pass */
void forward_pass_int4(int num_layers, int hidden, int intermediate,
                       float* hidden_state,
                       int4_tensor_t* W_up, int4_tensor_t* W_down) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int layer = 0; layer < num_layers; layer++) {
        rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
        matmul_int4_lut(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
        swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
        matmul_int4_lut(output_up, W_down, output_down, 1, hidden, intermediate);
        
        for (int j = 0; j < hidden; j++) {
            hidden_state[j] += output_down[j];
        }
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

/* INT8 baseline for comparison */
void forward_pass_int8(int num_layers, int hidden, int intermediate,
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
        
        for (int j = 0; j < hidden; j++) {
            hidden_state[j] += output_down[j];
        }
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  INT4 vs INT8: 32 LAYERS SHOWDOWN\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    
    /* Create float weights for INT4 */
    printf("Creating weights...\n");
    int up_size = 2 * intermediate * hidden;
    int down_size = hidden * intermediate;
    
    float* W_up_float = (float*)aligned_malloc(up_size * sizeof(float), 64);
    float* W_down_float = (float*)aligned_malloc(down_size * sizeof(float), 64);
    
    for (int i = 0; i < up_size; i++) {
        W_up_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < down_size; i++) {
        W_down_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Create INT4 tensors */
    printf("Quantizing to INT4...\n");
    int4_tensor_t* W_up_int4 = create_int4_tensor(W_up_float, 2*intermediate, hidden, 64);
    int4_tensor_t* W_down_int4 = create_int4_tensor(W_down_float, hidden, intermediate, 64);
    
    /* Create INT8 tensors for comparison */
    printf("Quantizing to INT8...\n");
    dequantized_tensor_t W_up_int8, W_down_int8;
    W_up_int8.rows = 2 * intermediate;
    W_up_int8.cols = hidden;
    W_up_int8.weights = (int8_t*)aligned_malloc(up_size, 64);
    W_up_int8.scales = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up_int8.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            float w = W_up_float[r * hidden + c];
            int q = (int)roundf(w / 0.01f);
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
            float w = W_down_float[r * intermediate + c];
            int q = (int)roundf(w / 0.01f);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            W_down_int8.weights[r * intermediate + c] = (int8_t)q;
        }
    }
    
    printf("\nMemory usage:\n");
    printf("  INT8: %.1f MB\n", (up_size + down_size) / (1024.0 * 1024.0));
    printf("  INT4: %.1f MB (%.1fx reduction)\n", 
           (up_size + down_size) / (2.0 * 1024.0 * 1024.0),
           2.0);
    
    /* Benchmark INT8 baseline */
    printf("\n\n1. INT8 BASELINE (32 layers)\n");
    printf("-----------------------------\n");
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        forward_pass_int8(num_layers, hidden, intermediate, hidden_state, &W_up_int8, &W_down_int8);
    }
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        forward_pass_int8(num_layers, hidden, intermediate, hidden_state, &W_up_int8, &W_down_int8);
    }
    double elapsed = get_time_ms() - start;
    double int8_speed = tokens / (elapsed / 1000.0);
    
    printf("   Speed: %.1f tok/sec\n", int8_speed);
    
    /* Benchmark INT4 */
    printf("\n2. INT4 OPTIMIZED (32 layers)\n");
    printf("------------------------------\n");
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        forward_pass_int4(num_layers, hidden, intermediate, hidden_state, W_up_int4, W_down_int4);
    }
    
    start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        forward_pass_int4(num_layers, hidden, intermediate, hidden_state, W_up_int4, W_down_int4);
    }
    elapsed = get_time_ms() - start;
    double int4_speed = tokens / (elapsed / 1000.0);
    
    printf("   Speed: %.1f tok/sec\n", int4_speed);
    
    double speedup = int4_speed / int8_speed;
    printf("\n   Speedup: %.2fx\n", speedup);
    
    /* Results */
    printf("\n========================================\n");
    printf("  RESULTS (32 LAYERS)\n");
    printf("========================================\n");
    printf("INT8: %.1f tok/sec\n", int8_speed);
    printf("INT4: %.1f tok/sec\n", int4_speed);
    printf("Target: 50 tok/sec\n\n");
    
    if (int4_speed >= 50.0) {
        printf("✅✅✅ 50 TOK/SEC ACHIEVED WITH INT4! ✅✅✅\n");
        printf("Margin: +%.1f%%\n", (int4_speed - 50.0) / 50.0 * 100.0);
    } else if (int4_speed > int8_speed) {
        double gap = 50.0 - int4_speed;
        printf("⚠️ Faster than INT8 but need %.1f more tok/sec\n", gap);
        printf("Current efficiency: %.0f%% of target\n", (int4_speed / 50.0) * 100.0);
    } else {
        printf("❌ INT4 dequant overhead too high\n");
        printf("Need more optimized unpacking\n");
    }
    
    printf("\n========================================\n");
    printf("  ANALYSIS\n");
    printf("========================================\n");
    printf("Memory bandwidth reduction: 2x\n");
    printf("Compute overhead: INT4 unpacking\n");
    printf("Net effect: %.1fx speedup\n\n", speedup);
    
    if (speedup < 1.5) {
        printf("Issue: Dequantization overhead negates bandwidth savings\n");
        printf("Solution needed: Faster INT4 unpacking with AVX2\n");
    }
    
    /* Cleanup */
    aligned_free(W_up_float);
    aligned_free(W_down_float);
    free_int4_tensor(W_up_int4);
    free_int4_tensor(W_down_int4);
    aligned_free(W_up_int8.weights);
    aligned_free(W_up_int8.scales);
    aligned_free(W_down_int8.weights);
    aligned_free(W_down_int8.scales);
    aligned_free(hidden_state);
    
    return 0;
}
