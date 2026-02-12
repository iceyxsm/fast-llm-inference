/*
 * Ultra-Optimized Benchmark
 * Tests the new 2-row parallel kernel
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

/* Declare ultra matmul */
extern void matmul_dequantized_ultra(const float* A, const dequantized_tensor_t* B,
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
    printf("  ULTRA-OPTIMIZED KERNEL BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    
    /* Create test weights */
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
    
    /* Input/output */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output = aligned_malloc(hidden * sizeof(float), 32);
    float* temp = aligned_malloc(2 * intermediate * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int w = 0; w < 10; w++) {
        matmul_dequantized_ultra(input, &gate_up, temp, 1, 2 * intermediate, hidden);
        matmul_dequantized_ultra(temp, &down, output, 1, hidden, intermediate);
    }
    
    /* Benchmark */
    printf("\nBenchmarking 1000 iterations...\n");
    double start = get_time_ms();
    
    for (int i = 0; i < 1000; i++) {
        /* Gate + Up */
        matmul_dequantized_ultra(input, &gate_up, temp, 1, 2 * intermediate, hidden);
        
        /* SwiGLU */
        for (int j = 0; j < intermediate; j++) {
            float g = temp[j];
            float u = temp[j + intermediate];
            float sig = 1.0f / (1.0f + expf(-g));
            temp[j] = g * sig * u;
        }
        
        /* Down */
        matmul_dequantized_ultra(temp, &down, output, 1, hidden, intermediate);
        
        /* Residual */
        for (int j = 0; j < hidden; j++) {
            output[j] += input[j];
        }
    }
    
    double elapsed = get_time_ms() - start;
    double ms_per_layer = elapsed;
    double tok_per_sec = 1000000.0 / (ms_per_layer * 32);  /* Scale to 32 layers */
    
    printf("\n=== RESULTS ===\n");
    printf("Time for 1000 iterations: %.2f ms\n", elapsed);
    printf("Time per layer (FFN): %.3f ms\n", ms_per_layer / 1000.0);
    printf("Estimated 32-layer speed: %.2f tok/sec\n", tok_per_sec);
    printf("\nvs llama.cpp (~25 tok/sec): %.2fx\n", tok_per_sec / 25.0);
    printf("\n");
    
    /* Cleanup */
    aligned_free(gate_up.weights);
    aligned_free(gate_up.scales);
    aligned_free(down.weights);
    aligned_free(down.scales);
    aligned_free(input);
    aligned_free(output);
    aligned_free(temp);
    
    return 0;
}
