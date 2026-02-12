/*
 * Quick benchmark - test only baseline vs VNNI
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#define HIDDEN_SIZE 3072
#define INTERMEDIATE_SIZE 8192
#define NUM_LAYERS 32

static double get_time(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* Dequantized tensor struct */
typedef struct {
    int8_t* weights;
    float* scales;
    int rows;
    int cols;
    int original_bits;
} dq_tensor_t;

/* External functions */
extern void matmul_dequantized_asm_style(const float* A, const dq_tensor_t* B,
                                          float* C, int M, int N, int K);

int main() {
    printf("Quick Benchmark - Baseline Only\n");
    printf("================================\n\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    
    printf("Allocating memory (%d x %d = %d MB)...\n", m, n, (m * n) / (1024 * 1024));
    
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    
    dq_tensor_t B;
    B.weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    B.scales = (float*)aligned_malloc(m * sizeof(float), 32);
    B.rows = m;
    B.cols = n;
    
    printf("Initializing...\n");
    for (int i = 0; i < n; i++) input[i] = 0.01f;
    for (int i = 0; i < m * n; i++) B.weights[i] = 1;
    for (int i = 0; i < m; i++) B.scales[i] = 0.01f;
    
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    
    printf("Benchmarking...\n");
    int iterations = 100;
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &B, output, 1, m, n);
    }
    double elapsed = get_time() - start;
    double avg_ms = (elapsed / iterations) * 1000;
    double tok_sec = 1.0 / ((elapsed / iterations) * NUM_LAYERS * 2);
    
    printf("\n=== RESULTS ===\n");
    printf("Iterations: %d\n", iterations);
    printf("Total time: %.2f ms\n", elapsed * 1000);
    printf("Avg per matmul: %.3f ms\n", avg_ms);
    printf("Tokens/sec: %.1f\n", tok_sec);
    printf("Target: 50 tok/sec\n");
    
    if (tok_sec >= 50) {
        printf("Status: ✓ TARGET ACHIEVED!\n");
    } else {
        printf("Status: ✗ Below target (need %.1fx more)\n", 50.0 / tok_sec);
    }
    
    aligned_free(input);
    aligned_free(output);
    aligned_free(B.weights);
    aligned_free(B.scales);
    
    return 0;
}
