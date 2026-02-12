/*
 * Final Optimized Benchmark
 * Best kernel + aligned memory + thread pool
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
#include "thread_pool.h"
#include "cpu_features.h"

extern void matmul_dequantized_best(const float* A, const dequantized_tensor_t* B,
                                     float* C, int M, int N, int K);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  FINAL OPTIMIZED BENCHMARK\n");
    printf("  Target: 25 tok/sec\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    int num_tokens = 100;
    
    /* Create thread pool */
    thread_pool_t* pool = thread_pool_create(cpu.num_cores);
    
    /* Create weights with 64-byte alignment */
    dequantized_tensor_t gate_up;
    gate_up.rows = 2 * intermediate;
    gate_up.cols = hidden;
    gate_up.weights = aligned_malloc(2 * intermediate * hidden, 64);
    gate_up.scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        gate_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            gate_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    dequantized_tensor_t down;
    down.rows = hidden;
    down.cols = intermediate;
    down.weights = aligned_malloc(hidden * intermediate, 64);
    down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < hidden; r++) {
        down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    /* Aligned buffers */
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* output = aligned_malloc(hidden * sizeof(float), 64);
    float* temp = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int w = 0; w < 10; w++) {
        matmul_dequantized_best(input, &gate_up, temp, 1, 2 * intermediate, hidden);
        matmul_dequantized_best(temp, &down, output, 1, hidden, intermediate);
    }
    
    /* Benchmark */
    printf("\nBenchmarking %d tokens with %d layers...\n", num_tokens, num_layers);
    double start = get_time_ms();
    
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < num_layers; layer++) {
            /* Gate + Up */
            matmul_dequantized_best(input, &gate_up, temp, 1, 2 * intermediate, hidden);
            
            /* SwiGLU */
            for (int j = 0; j < intermediate; j++) {
                float g = temp[j];
                float u = temp[j + intermediate];
                float sig = 1.0f / (1.0f + expf(-g));
                temp[j] = g * sig * u;
            }
            
            /* Down */
            matmul_dequantized_best(temp, &down, output, 1, hidden, intermediate);
            
            /* Residual */
            for (int j = 0; j < hidden; j++) {
                output[j] += input[j];
            }
            
            float* t = input; input = output; output = t;
        }
        
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    double elapsed = get_time_ms() - start;
    double tok_per_sec = num_tokens / (elapsed / 1000.0);
    
    printf("\n\n=== FINAL RESULTS ===\n");
    printf("Configuration: %d layers, %d hidden, %d intermediate\n", num_layers, hidden, intermediate);
    printf("Tokens: %d\n", num_tokens);
    printf("Time: %.2f ms\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("Ms/token: %.2f\n", elapsed / num_tokens);
    printf("\n");
    printf("TARGET: 25 tok/sec\n");
    printf("ACHIEVED: %.2f tok/sec\n", tok_per_sec);
    printf("GAP: %.1f%% of target\n", (tok_per_sec / 25.0) * 100.0);
    printf("\n");
    
    if (tok_per_sec >= 25.0) {
        printf("🎉 TARGET ACHIEVED! 🎉\n");
    } else if (tok_per_sec >= 20.0) {
        printf("⚠️ Close to target (80%+)\n");
    } else {
        printf("❌ Need more optimization\n");
    }
    printf("\n");
    
    /* Cleanup */
    thread_pool_destroy(pool);
    aligned_free(gate_up.weights);
    aligned_free(gate_up.scales);
    aligned_free(down.weights);
    aligned_free(down.scales);
    aligned_free(input);
    aligned_free(output);
    aligned_free(temp);
    
    return 0;
}
