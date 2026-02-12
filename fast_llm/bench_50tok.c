/*
 * Benchmark for 50 tok/sec Target
 * Tests new 12x16 micro-kernel
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
extern void matmul_dequantized_50tok(const float* A, const dequantized_tensor_t* B,
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
    double tok_per_sec = 1000.0 / (ms_per * 32 * 2);  /* 32 layers, 2 matmuls */
    
    printf("%-20s: %6.3f ms | %5.1f GFLOPS | %5.2f tok/sec\n", 
           name, ms_per, gflops, tok_per_sec);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  50 TOK/SEC TARGET BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 200;
    
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
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Gate+Up Projection [%d, %d]:\n", 2*intermediate, hidden);
    benchmark_kernel("6x16 (30 tok baseline)", matmul_dequantized_asm_style,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    benchmark_kernel("12x16 (50 tok target)", matmul_dequantized_50tok,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, iterations);
    
    printf("\nDown Projection [%d, %d]:\n", hidden, intermediate);
    benchmark_kernel("6x16 (30 tok baseline)", matmul_dequantized_asm_style,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    benchmark_kernel("12x16 (50 tok target)", matmul_dequantized_50tok,
                     output_up, &W_down, output_down, 1, hidden, intermediate, iterations);
    
    /* Full model test with best kernel */
    printf("\n========================================\n");
    printf("  FULL MODEL TEST (12x16 kernel)\n");
    printf("========================================\n\n");
    
    int num_tokens = 100;
    printf("Running %d tokens with 32 layers...\n", num_tokens);
    
    double start = get_time_ms();
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < 32; layer++) {
            /* Gate + Up */
            matmul_dequantized_50tok(input, &W_up, output_up, 1, 2*intermediate, hidden);
            
            /* SwiGLU */
            for (int j = 0; j < intermediate; j++) {
                float g = output_up[j];
                float u = output_up[j + intermediate];
                float sig = 1.0f / (1.0f + expf(-g));
                output_up[j] = g * sig * u;
            }
            
            /* Down */
            matmul_dequantized_50tok(output_up, &W_down, output_down, 1, hidden, intermediate);
            
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
    
    printf("\n\n=== RESULTS ===\n");
    printf("Time: %.2f ms\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("TARGET: 50 tok/sec\n");
    printf("GAP: %.1f%% of target\n", (tok_per_sec / 50.0) * 100.0);
    printf("\n");
    
    if (tok_per_sec >= 50.0) {
        printf("🎉🎉🎉 50 TOK/SEC ACHIEVED! 🎉🎉🎉\n");
    } else if (tok_per_sec >= 40.0) {
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
