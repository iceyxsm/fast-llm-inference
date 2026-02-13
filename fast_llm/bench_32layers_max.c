/*
 * Maximum Performance WITH 32 LAYERS
 * No layer reduction - pure optimization only
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
extern void matmul_dequantized_large_n(const float* A, const dequantized_tensor_t* B,
                                        float* C, int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Version 1: Baseline 32 layers */
double benchmark_v1_baseline(int num_layers, int hidden, int intermediate,
                              dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* hidden_state = aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        for (int layer = 0; layer < num_layers; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    
    /* Benchmark */
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

/* Version 2: In-place RMS norm (reuse buffers) */
double benchmark_v2_inplace(int num_layers, int hidden, int intermediate,
                             dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* hidden_state = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            /* In-place RMS norm */
            rms_norm_avx2(hidden_state, hidden_state, hidden, 1e-5f);
            matmul_dequantized_asm_style(hidden_state, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

/* Version 3: Use large-N kernel for gate+up projection */
double benchmark_v3_mixed_kernels(int num_layers, int hidden, int intermediate,
                                   dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* hidden_state = aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Determine best kernel for large N (gate+up) */
    double best_time = 1e9;
    void (*best_up_kernel)(const float*, const dequantized_tensor_t*, float*, int, int, int) = matmul_dequantized_asm_style;
    
    /* Quick test */
    double start_test = get_time_ms();
    for (int i = 0; i < 20; i++) {
        matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
    }
    double t1 = (get_time_ms() - start_test) / 20.0;
    
    start_test = get_time_ms();
    for (int i = 0; i < 20; i++) {
        matmul_dequantized_large_n(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
    }
    double t2 = (get_time_ms() - start_test) / 20.0;
    
    if (t2 < t1) best_up_kernel = matmul_dequantized_large_n;
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            best_up_kernel(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

/* Version 4: Unroll the layer loop (compiler hint) */
double benchmark_v4_unrolled(int num_layers, int hidden, int intermediate,
                              dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* hidden_state = aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        int layer = 0;
        /* Unroll by 4 */
        for (; layer + 3 < num_layers; layer += 4) {
            for (int k = 0; k < 4; k++) {
                rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
                matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
                swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
                matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
                for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
            }
        }
        /* Remaining layers */
        for (; layer < num_layers; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

/* Version 5: Layer parallelism (process multiple layers in parallel per token) */
double benchmark_v5_layer_parallel(int num_layers, int hidden, int intermediate,
                                    dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    /* This is tricky for autoregressive but we can try pipelining */
    /* For now, just use best single-threaded version */
    return benchmark_v1_baseline(num_layers, hidden, intermediate, W_up, W_down);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  32 LAYERS - NO REDUCTION ALLOWED\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    
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
    
    printf("Testing different 32-layer optimizations...\n\n");
    
    /* Test all versions */
    printf("V1 - Baseline:          "); fflush(stdout);
    double v1 = benchmark_v1_baseline(num_layers, hidden, intermediate, &W_up, &W_down);
    printf("%.1f tok/sec\n", v1);
    
    printf("V2 - In-place RMSNorm:  "); fflush(stdout);
    double v2 = benchmark_v2_inplace(num_layers, hidden, intermediate, &W_up, &W_down);
    printf("%.1f tok/sec\n", v2);
    
    printf("V3 - Mixed kernels:     "); fflush(stdout);
    double v3 = benchmark_v3_mixed_kernels(num_layers, hidden, intermediate, &W_up, &W_down);
    printf("%.1f tok/sec\n", v3);
    
    printf("V4 - Unrolled layers:   "); fflush(stdout);
    double v4 = benchmark_v4_unrolled(num_layers, hidden, intermediate, &W_up, &W_down);
    printf("%.1f tok/sec\n", v4);
    
    printf("\n----------------------------------------\n");
    printf("TARGET: 50 tok/sec with 32 layers\n");
    printf("----------------------------------------\n\n");
    
    double best = v1;
    if (v2 > best) best = v2;
    if (v3 > best) best = v3;
    if (v4 > best) best = v4;
    
    printf("Best result: %.1f tok/sec\n", best);
    
    if (best >= 50.0) {
        printf("✅ 32 LAYERS TARGET ACHIEVED!\n");
    } else {
        double gap = (50.0 - best) / 50.0 * 100.0;
        printf("❌ Gap: %.1f%% (need %.1f more tok/sec)\n", gap, 50.0 - best);
        printf("\nNeed more aggressive optimizations:\n");
        printf("- INT4 quantization\n");
        printf("- Custom ASM kernels\n");
        printf("- Better memory prefetching\n");
    }
    
    /* Cleanup */
    aligned_free(W_up.weights);
    aligned_free(W_up.scales);
    aligned_free(W_down.weights);
    aligned_free(W_down.scales);
    
    return 0;
}
