/*
 * Fused Kernel Benchmark
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
extern void matmul_gate_up_swiglu_fused(const float* A,
                                         const dequantized_tensor_t* B_gate,
                                         const dequantized_tensor_t* B_up,
                                         float* C,
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
    printf("  FUSED KERNEL BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    
    /* Create weights - separate gate and up matrices */
    dequantized_tensor_t W_gate, W_up, W_down;
    
    /* Gate: [inter, hidden] = [8192, 3072] */
    W_gate.rows = intermediate;
    W_gate.cols = hidden;
    W_gate.weights = aligned_malloc(intermediate * hidden, 64);
    W_gate.scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    /* Up: [inter, hidden] = [8192, 3072] */
    W_up.rows = intermediate;
    W_up.cols = hidden;
    W_up.weights = aligned_malloc(intermediate * hidden, 64);
    W_up.scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    /* Down: [hidden, intermediate] = [3072, 8192] */
    W_down.rows = hidden;
    W_down.cols = intermediate;
    W_down.weights = aligned_malloc(hidden * intermediate, 64);
    W_down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < intermediate; r++) {
        W_gate.scales[r] = 0.01f;
        W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_gate.weights[r * hidden + c] = (rand() % 256) - 128;
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    for (int r = 0; r < hidden; r++) {
        W_down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            W_down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* gate_out = aligned_malloc(intermediate * sizeof(float), 64);
    float* up_out = aligned_malloc(intermediate * sizeof(float), 64);
    float* swiglu_out = aligned_malloc(intermediate * sizeof(float), 64);
    float* final_out = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    int iterations = 100;
    
    /* Benchmark separate vs fused */
    printf("Gate+Up+SwiGLU Benchmark:\n");
    printf("(Intermediate = %d)\n\n", intermediate);
    
    /* Separate approach */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &W_gate, gate_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(input, &W_up, up_out, 1, intermediate, hidden);
        /* SwiGLU */
        for (int j = 0; j < intermediate; j++) {
            float g = gate_out[j];
            float u = up_out[j];
            float sig = 1.0f / (1.0f + expf(-g));
            swiglu_out[j] = g * sig * u;
        }
    }
    double t_separate = (get_time_ms() - start) / iterations;
    printf("Separate (2x matmul + SwiGLU): %.3f ms\n", t_separate);
    
    /* Fused approach */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_gate_up_swiglu_fused(input, &W_gate, &W_up, swiglu_out, 1, intermediate, hidden);
    }
    double t_fused = (get_time_ms() - start) / iterations;
    printf("Fused (matmul+SwiGLU):         %.3f ms\n", t_fused);
    
    printf("\nSpeedup: %.2fx\n", t_separate / t_fused);
    
    /* Full model estimate */
    start = get_time_ms();
    for (int i = 0; i < 50; i++) {
        matmul_dequantized_asm_style(swiglu_out, &W_down, final_out, 1, hidden, intermediate);
    }
    double t_down = (get_time_ms() - start) / 50.0;
    printf("\nDown projection: %.3f ms\n", t_down);
    
    double t_total_fused = t_fused + t_down;
    double tok_sec = 1000.0 / t_total_fused / 32.0;
    
    printf("\n=== ESTIMATED PERFORMANCE ===\n");
    printf("Per layer: %.3f ms\n", t_total_fused);
    printf("Tokens/sec: %.2f\n", tok_sec);
    printf("Target: 50 tok/sec\n");
    printf("Gap: %.1f%%\n", (tok_sec / 50.0) * 100.0);
    
    /* Cleanup */
    aligned_free(W_gate.weights); aligned_free(W_gate.scales);
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(input); aligned_free(gate_out); aligned_free(up_out);
    aligned_free(swiglu_out); aligned_free(final_out);
    
    return 0;
}
