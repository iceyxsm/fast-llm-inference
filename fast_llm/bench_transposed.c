/*
 * Benchmark transposed weights matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

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
extern void transpose_weights_16xN(const int8_t* src, int N, int K,
                                    int8_t* dst, const float* src_scales, float* dst_scales);
extern void matmul_dequantized_transposed(const float* A, const int8_t* B_t,
                                           const float* scales_B,
                                           float* C, int M, int N, int K);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

void benchmark_kernel(const char* name, double ms, double gflops) {
    printf("%-30s: %6.3f ms | %5.1f GFLOPS\n", name, ms, gflops);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  TRANSPOSED WEIGHTS BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 100;
    
    /* Create standard weights */
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
    
    /* Create transposed weights */
    int8_t* W_up_t = aligned_malloc(2 * intermediate * hidden, 64);
    float* W_up_t_scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    transpose_weights_16xN(W_up.weights, 2*intermediate, hidden, W_up_t, W_up.scales, W_up_t_scales);
    
    int8_t* W_down_t = aligned_malloc(hidden * intermediate, 64);
    float* W_down_t_scales = aligned_malloc(hidden * sizeof(float), 64);
    transpose_weights_16xN(W_down.weights, hidden, intermediate, W_down_t, W_down.scales, W_down_t_scales);
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* output = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Gate+Up (%d x %d):\n\n", 2*intermediate, hidden);
    
    /* Benchmark standard */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &W_up, output, 1, 2*intermediate, hidden);
    }
    double t_std = (get_time_ms() - start) / iterations;
    benchmark_kernel("Standard [N,K]", t_std, (2.0*2*intermediate*hidden)/(t_std*1e6));
    
    /* Benchmark transposed */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_transposed(input, W_up_t, W_up_t_scales, output, 1, 2*intermediate, hidden);
    }
    double t_trans = (get_time_ms() - start) / iterations;
    benchmark_kernel("Transposed [K/16,N,16]", t_trans, (2.0*2*intermediate*hidden)/(t_trans*1e6));
    
    printf("\nSpeedup: %.2fx\n", t_std / t_trans);
    
    printf("\nDown (%d x %d):\n\n", hidden, intermediate);
    
    /* Benchmark standard */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(output, &W_down, input, 1, hidden, intermediate);
    }
    t_std = (get_time_ms() - start) / iterations;
    benchmark_kernel("Standard [N,K]", t_std, (2.0*hidden*intermediate)/(t_std*1e6));
    
    /* Benchmark transposed */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_transposed(output, W_down_t, W_down_t_scales, input, 1, hidden, intermediate);
    }
    t_trans = (get_time_ms() - start) / iterations;
    benchmark_kernel("Transposed [K/16,N,16]", t_trans, (2.0*hidden*intermediate)/(t_trans*1e6));
    
    printf("\nSpeedup: %.2fx\n", t_std / t_trans);
    
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(W_up_t); aligned_free(W_up_t_scales);
    aligned_free(W_down_t); aligned_free(W_down_t_scales);
    aligned_free(input); aligned_free(output);
    
    return 0;
}
