/*
 * Test AVX2 Optimizations in Full Model Context
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

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

extern void matmul_dequantized_asm_style(const float* A, void* B, float* C,
                                          int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

static double get_time(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

typedef struct {
    int8_t* weights;
    float* scales;
} sim_tensor_t;

int main() {
    printf("Testing AVX2 Optimizations in Full Model\n");
    printf("=========================================\n\n");
    
    int hidden = HIDDEN_SIZE;
    int intermediate = INTERMEDIATE_SIZE;
    
    /* Setup weights */
    sim_tensor_t gate_proj, up_proj, down_proj;
    gate_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate * hidden, 32);
    up_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate * hidden, 32);
    down_proj.weights = (int8_t*)aligned_malloc((size_t)hidden * intermediate, 32);
    gate_proj.scales = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    up_proj.scales = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    down_proj.scales = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    for (int i = 0; i < intermediate * hidden; i++) {
        gate_proj.weights[i] = (int8_t)(i % 256 - 128);
        up_proj.weights[i] = (int8_t)(i % 256 - 128);
    }
    for (int i = 0; i < hidden * intermediate; i++) {
        down_proj.weights[i] = (int8_t)(i % 256 - 128);
    }
    for (int i = 0; i < intermediate; i++) {
        gate_proj.scales[i] = up_proj.scales[i] = 0.01f;
    }
    for (int i = 0; i < hidden; i++) {
        down_proj.scales[i] = 0.01f;
    }
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* gate_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    printf("Running 50 tokens with 32 layers...\n");
    int iterations = 50;
    
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            /* RMS Norm */
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            
            /* Gate projection */
            matmul_dequantized_asm_style(norm_out, &gate_proj, gate_out, 1, intermediate, hidden);
            
            /* Up projection */
            matmul_dequantized_asm_style(norm_out, &up_proj, up_out, 1, intermediate, hidden);
            
            /* SwiGLU - AVX2 optimized */
            swiglu_avx2(gate_out, up_out, gate_out, intermediate);
            
            /* Down projection */
            matmul_dequantized_asm_style(gate_out, &down_proj, hidden_state, 1, hidden, intermediate);
        }
        if ((iter + 1) % 10 == 0) {
            printf("  %d/%d\n", iter + 1, iterations);
        }
    }
    double elapsed = get_time() - start;
    double tok_sec = iterations / elapsed;
    
    printf("\nResults:\n");
    printf("  Time: %.2f seconds\n", elapsed);
    printf("  Speed: %.1f tokens/second\n", tok_sec);
    
    if (tok_sec >= 50.0) {
        printf("  ✅ TARGET ACHIEVED with AVX2 optimizations!\n");
    } else {
        printf("  ❌ Below target\n");
    }
    
    /* Cleanup */
    aligned_free(gate_proj.weights);
    aligned_free(up_proj.weights);
    aligned_free(down_proj.weights);
    aligned_free(gate_proj.scales);
    aligned_free(up_proj.scales);
    aligned_free(down_proj.scales);
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(gate_out);
    aligned_free(up_out);
    
    return 0;
}
