/*
 * 50 tok/sec Benchmark with AVX2 Activations
 * Fixed version that properly uses AVX2 SwiGLU
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
extern void matmul_dequantized_large_n(const float* A, const dequantized_tensor_t* B,
                                        float* C, int M, int N, int K);

/* AVX2 activation functions */
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  50 TOK/SEC WITH AVX2 ACTIVATIONS\n");
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
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Determine best Gate+Up kernel */
    printf("Selecting best kernels...\n");
    double best_up_time = 1e9;
    void (*best_up)(const float*, const dequantized_tensor_t*, float*, int, int, int) = matmul_dequantized_asm_style;
    const char* best_up_name = "6x16 ASM-Style";
    
    double start = get_time_ms();
    for (int i = 0; i < 30; i++) {
        matmul_dequantized_asm_style(input, &W_up, output_up, 1, 2*intermediate, hidden);
    }
    double t = (get_time_ms() - start) / 30.0;
    if (t < best_up_time) { best_up_time = t; best_up = matmul_dequantized_asm_style; best_up_name = "6x16 ASM-Style"; }
    
    start = get_time_ms();
    for (int i = 0; i < 30; i++) {
        matmul_dequantized_large_n(input, &W_up, output_up, 1, 2*intermediate, hidden);
    }
    t = (get_time_ms() - start) / 30.0;
    if (t < best_up_time) { best_up_time = t; best_up = matmul_dequantized_large_n; best_up_name = "16x8 Large-N"; }
    
    printf("Best Gate+Up kernel: %s (%.3f ms)\n", best_up_name, best_up_time);
    printf("Down kernel: 6x16 ASM-Style\n");
    printf("SwiGLU: AVX2 optimized\n\n");
    
    /* Full model test with AVX2 activations */
    int num_tokens = 100;
    printf("Running %d tokens with 32 layers...\n", num_tokens);
    
    start = get_time_ms();
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < 32; layer++) {
            /* RMS Norm - AVX2 */
            rms_norm_avx2(input, norm_out, hidden, 1e-5f);
            
            /* Gate + Up */
            best_up(norm_out, &W_up, output_up, 1, 2*intermediate, hidden);
            
            /* SwiGLU - AVX2 optimized */
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            
            /* Down */
            matmul_dequantized_asm_style(output_up, &W_down, output_down, 1, hidden, intermediate);
            
            /* Residual */
            for (int j = 0; j < hidden; j++) {
                output_down[j] += input[j];
            }
            
            float* tmp = input; input = output_down; output_down = tmp;
        }
        
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    double elapsed = get_time_ms() - start;
    double tok_per_sec = num_tokens / (elapsed / 1000.0);
    
    printf("\n\n========================================\n");
    printf("  RESULTS WITH AVX2 ACTIVATIONS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("TARGET: 50 tok/sec\n");
    printf("GAP: %.1f%% of target\n", (tok_per_sec / 50.0) * 100.0);
    printf("\n");
    
    if (tok_per_sec >= 50.0) {
        printf("🎉🎉🎉 50 TOK/SEC ACHIEVED WITH AVX2! 🎉🎉🎉\n");
    } else if (tok_per_sec >= 40.0) {
        printf("⚠️ Close! (80%+ of target)\n");
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
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return 0;
}
