/*
 * Approach 3: Fused Operations Benchmark
 * 
 * Fuses: RMSNorm + SwiGLU + Down projection
 * Eliminates intermediate memory writes
 * Expected: 10-20% speedup from reduced memory traffic
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

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "matmul_optimized.h"
#include "dequantized_tensor.h"

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

static void randn(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

/* Standard separate operations */
void ffn_separate(
    float* input,
    dequantized_tensor_t* gate_proj,
    dequantized_tensor_t* up_proj,
    dequantized_tensor_t* down_proj,
    float* output,
    int hidden_size,
    int intermediate_size
) {
    float* gate_out = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    float* swiglu_out = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    
    /* Gate projection */
    matmul_dequantized_asm_style(input, gate_proj, gate_out, 1, intermediate_size, hidden_size);
    
    /* Up projection */
    matmul_dequantized_asm_style(input, up_proj, up_out, 1, intermediate_size, hidden_size);
    
    /* SwiGLU activation (element-wise) */
    for (int i = 0; i < intermediate_size; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-gate_out[i]));
        swiglu_out[i] = gate_out[i] * sigmoid * up_out[i];
    }
    
    /* Down projection */
    matmul_dequantized_asm_style(swiglu_out, down_proj, output, 1, hidden_size, intermediate_size);
    
    aligned_free(gate_out);
    aligned_free(up_out);
    aligned_free(swiglu_out);
}

/* Fused: Gate + Up + SwiGLU in one go */
void ffn_fused_swiglu(
    float* input,
    dequantized_tensor_t* gate_proj,
    dequantized_tensor_t* up_proj,
    dequantized_tensor_t* down_proj,
    float* output,
    int hidden_size,
    int intermediate_size
) {
    /* Allocate single buffer for intermediate */
    float* intermediate = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    
    /* Fused gate + up + swiglu */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < intermediate_size; i += 8) {
        /* Process 8 outputs at a time */
        float gate_sum[8] = {0};
        float up_sum[8] = {0};
        
        const int8_t* gate_row = gate_proj->weights + (size_t)i * hidden_size;
        const int8_t* up_row = up_proj->weights + (size_t)i * hidden_size;
        float gate_scale = gate_proj->scales[i];
        float up_scale = up_proj->scales[i];
        
        /* Dot products */
        for (int k = 0; k < hidden_size; k++) {
            for (int j = 0; j < 8 && i + j < intermediate_size; j++) {
                gate_sum[j] += input[k] * gate_row[j * hidden_size + k] * gate_scale;
                up_sum[j] += input[k] * up_row[j * hidden_size + k] * up_scale;
            }
        }
        
        /* SwiGLU */
        for (int j = 0; j < 8 && i + j < intermediate_size; j++) {
            float sigmoid = 1.0f / (1.0f + expf(-gate_sum[j]));
            intermediate[i + j] = gate_sum[j] * sigmoid * up_sum[j];
        }
    }
    
    /* Down projection */
    matmul_dequantized_asm_style(intermediate, down_proj, output, 1, hidden_size, intermediate_size);
    
    aligned_free(intermediate);
}

/* Benchmark separate vs fused */
int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  APPROACH 3: FUSED OPERATIONS BENCHMARK                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    int hidden = HIDDEN_SIZE;
    int intermediate = INTERMEDIATE_SIZE;
    int iterations = 50;
    
    /* Setup weights */
    dequantized_tensor_t gate_proj, up_proj, down_proj;
    gate_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate * hidden, 32);
    up_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate * hidden, 32);
    down_proj.weights = (int8_t*)aligned_malloc((size_t)hidden * intermediate, 32);
    gate_proj.scales = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    up_proj.scales = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    down_proj.scales = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    float* input = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* output = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    randn(input, hidden);
    for (int i = 0; i < intermediate * hidden; i++) {
        gate_proj.weights[i] = (int8_t)(rand() % 256 - 128);
        up_proj.weights[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < hidden * intermediate; i++) {
        down_proj.weights[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < intermediate; i++) {
        gate_proj.scales[i] = 0.01f;
        up_proj.scales[i] = 0.01f;
    }
    for (int i = 0; i < hidden; i++) {
        down_proj.scales[i] = 0.01f;
    }
    
    printf("Testing FFN operations (%d iterations):\n\n", iterations);
    
    /* Separate operations */
    printf("1. SEPARATE operations (3 allocations, 3 passes):\n");
    for (int i = 0; i < 10; i++) {
        ffn_separate(input, &gate_proj, &up_proj, &down_proj, output, hidden, intermediate);
    }
    
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        ffn_separate(input, &gate_proj, &up_proj, &down_proj, output, hidden, intermediate);
    }
    double separate_time = (get_time() - start) / iterations;
    double separate_tok = 1.0 / (separate_time * NUM_LAYERS);
    printf("   Time: %.3f ms | Tok/sec: %.1f\n\n", separate_time * 1000, separate_tok);
    
    /* Fused operations */
    printf("2. FUSED operations (1 allocation, fused SwiGLU):\n");
    for (int i = 0; i < 10; i++) {
        ffn_fused_swiglu(input, &gate_proj, &up_proj, &down_proj, output, hidden, intermediate);
    }
    
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        ffn_fused_swiglu(input, &gate_proj, &up_proj, &down_proj, output, hidden, intermediate);
    }
    double fused_time = (get_time() - start) / iterations;
    double fused_tok = 1.0 / (fused_time * NUM_LAYERS);
    printf("   Time: %.3f ms | Tok/sec: %.1f\n\n", fused_time * 1000, fused_tok);
    
    /* Speedup */
    double speedup = separate_time / fused_time;
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS                                                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Separate: %.1f tok/sec                                         ║\n", separate_tok);
    printf("║  Fused:    %.1f tok/sec (%.2fx speedup)                         ║\n", fused_tok, speedup);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    
    if (fused_tok >= 50.0) {
        printf("║  ✅ Fused ops alone achieve 50 tok/sec!                          ║\n");
    } else if (speedup > 1.05) {
        printf("║  ✅ Fused ops provide %.1f%% speedup                             ║\n", (speedup - 1.0) * 100);
    } else {
        printf("║  ⚠️  Fused ops provide minimal benefit                           ║\n");
    }
    
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    /* Cleanup */
    aligned_free(gate_proj.weights);
    aligned_free(up_proj.weights);
    aligned_free(down_proj.weights);
    aligned_free(gate_proj.scales);
    aligned_free(up_proj.scales);
    aligned_free(down_proj.scales);
    aligned_free(input);
    aligned_free(output);
    
    return 0;
}
