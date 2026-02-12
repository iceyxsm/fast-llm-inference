/*
 * Native Q4_K Benchmark
 * Compares dequantized INT8 vs native Q4_K matmul
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
#include "ggml_quants.h"
#include "cpu_features.h"

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Create test Q4_K weights */
void create_test_q4k(void* weights, int rows, int cols) {
    int n = rows * cols;
    int nb = n / 256;
    block_q4_K* blocks = (block_q4_K*)weights;
    
    for (int b = 0; b < nb; b++) {
        /* Random scales */
        blocks[b].scales[0] = 10 + (rand() % 50);
        blocks[b].scales[1] = 5 + (rand() % 20);
        
        /* Random 4-bit values */
        for (int i = 0; i < 128; i++) {
            blocks[b].qs[i] = (rand() & 0x0F) | ((rand() & 0x0F) << 4);
        }
    }
}

/* Convert Q4_K to dequantized int8 */
void q4k_to_int8(const void* q4k_weights, dequantized_tensor_t* dt, int rows, int cols) {
    const block_q4_K* blocks = (const block_q4_K*)q4k_weights;
    int nb = (rows * cols) / 256;
    
    dt->rows = rows;
    dt->cols = cols;
    dt->weights = aligned_malloc(rows * cols, 64);
    dt->scales = aligned_malloc(rows * sizeof(float), 64);
    
    /* Approximate dequantization */
    for (int r = 0; r < rows; r++) {
        int block_idx = (r * cols / 256);  /* Simplified - assumes 1 scale per row */
        float d = ((blocks[block_idx].scales[0] & 0x3F) + 1) / 64.0f;
        float m = ((blocks[block_idx].scales[1] & 0x3F) + 1) / 64.0f;
        
        /* Find max for scale */
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            int b_idx = (r * cols + c) / 256;
            int offset = ((r * cols + c) % 256) / 2;
            int is_high = ((r * cols + c) % 2);
            
            uint8_t qv = blocks[b_idx].qs[offset];
            int val = is_high ? (qv >> 4) : (qv & 0x0F);
            float w = val * d + m;
            if (fabsf(w) > max_abs) max_abs = fabsf(w);
        }
        
        dt->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
        float inv = 1.0f / dt->scales[r];
        
        for (int c = 0; c < cols; c++) {
            int b_idx = (r * cols + c) / 256;
            int offset = ((r * cols + c) % 256) / 2;
            int is_high = ((r * cols + c) % 2);
            
            uint8_t qv = blocks[b_idx].qs[offset];
            int val = is_high ? (qv >> 4) : (qv & 0x0F);
            float w = val * d + m;
            
            int int8_val = (int)(w * inv);
            if (int8_val > 127) int8_val = 127;
            if (int8_val < -128) int8_val = -128;
            dt->weights[r * cols + c] = (int8_t)int8_val;
        }
    }
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  NATIVE Q4_K vs INT8 BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    /* Phi-3 dimensions */
    int hidden = 3072;
    int intermediate = 8192;
    int n = hidden * intermediate;
    int nb = n / 256;
    
    printf("Matrix: [%d, %d] = %d elements, %d Q4_K blocks\n\n", 
           intermediate, hidden, n, nb);
    
    /* Allocate weights */
    void* q4k_weights = aligned_malloc(nb * sizeof(block_q4_K), 64);
    create_test_q4k(q4k_weights, intermediate, hidden);
    
    /* Convert to int8 for comparison */
    dequantized_tensor_t dt;
    printf("Converting Q4_K to INT8...\n");
    q4k_to_int8(q4k_weights, &dt, intermediate, hidden);
    printf("Done. INT8 size: %.1f MB\n", (dt.rows * dt.cols) / (1024.0 * 1024.0));
    printf("Q4_K size: %.1f MB (%.1fx smaller)\n\n", 
           (nb * sizeof(block_q4_K)) / (1024.0 * 1024.0),
           (float)(dt.rows * dt.cols) / (nb * sizeof(block_q4_K)));
    
    /* Input/output */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output_q4k = aligned_malloc(intermediate * sizeof(float), 32);
    float* output_int8 = aligned_malloc(intermediate * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int i = 0; i < 10; i++) {
        ggml_gemv_q4_K(hidden, intermediate, output_q4k, q4k_weights, input);
        matmul_dequantized(input, &dt, output_int8, 1, intermediate, hidden);
    }
    
    /* Benchmark native Q4_K */
    printf("\nBenchmarking native Q4_K...\n");
    double start = get_time_ms();
    for (int i = 0; i < 100; i++) {
        ggml_gemv_q4_K(hidden, intermediate, output_q4k, q4k_weights, input);
    }
    double q4k_time = get_time_ms() - start;
    
    /* Benchmark dequantized INT8 */
    printf("Benchmarking dequantized INT8...\n");
    start = get_time_ms();
    for (int i = 0; i < 100; i++) {
        matmul_dequantized(input, &dt, output_int8, 1, intermediate, hidden);
    }
    double int8_time = get_time_ms() - start;
    
    /* Results */
    printf("\n=== RESULTS ===\n");
    printf("Native Q4_K:  %.3f ms (%.2f GFLOPS)\n", 
           q4k_time / 100.0, 
           (2.0 * hidden * intermediate * 100) / (q4k_time * 1e6));
    printf("INT8:         %.3f ms (%.2f GFLOPS)\n", 
           int8_time / 100.0,
           (2.0 * hidden * intermediate * 100) / (int8_time * 1e6));
    printf("Speedup:      %.2fx\n", int8_time / q4k_time);
    
    /* Estimate full model speed */
    /* 32 layers, 2 matmuls per layer (up + down) for FFN only */
    double ms_per_token_q4k = (q4k_time / 100.0) * 32 * 2;
    double tok_per_sec_q4k = 1000.0 / ms_per_token_q4k;
    
    double ms_per_token_int8 = (int8_time / 100.0) * 32 * 2;
    double tok_per_sec_int8 = 1000.0 / ms_per_token_int8;
    
    printf("\nEstimated 32-layer FFN-only speed:\n");
    printf("  Q4_K: %.2f tok/sec\n", tok_per_sec_q4k);
    printf("  INT8: %.2f tok/sec\n", tok_per_sec_int8);
    printf("\n");
    
    aligned_free(q4k_weights);
    aligned_free(dt.weights);
    aligned_free(dt.scales);
    aligned_free(input);
    aligned_free(output_q4k);
    aligned_free(output_int8);
    
    return 0;
}
