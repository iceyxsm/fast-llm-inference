/*
 * Benchmark: AVX2 Optimized Activations vs Scalar
 * 
 * Target: Reduce activation overhead from 53% to <20%
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

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

/* External AVX2 functions */
extern void silu_avx2(const float* input, float* output, int n);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

/* Scalar versions */
void silu_scalar(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void swiglu_scalar(const float* gate, const float* up, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        output[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

void rms_norm_scalar(const float* input, float* output, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * scale;
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  APPROACH 4: AVX2-Optimized Activations                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    int n = INTERMEDIATE_SIZE;
    int iterations = 1000;
    
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* input2 = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(n * sizeof(float), 32);
    
    for (int i = 0; i < n; i++) {
        input[i] = (float)rand() / RAND_MAX - 0.5f;
        input2[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    printf("Benchmarking %d elements (%d iterations):\n\n", n, iterations);
    
    /* SiLU */
    printf("1. SiLU (x * sigmoid(x)):\n");
    
    /* Scalar */
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        silu_scalar(input, output, n);
    }
    double silu_scalar_time = (get_time() - start) / iterations;
    printf("   Scalar: %.4f ms\n", silu_scalar_time * 1000);
    
    /* AVX2 */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        silu_avx2(input, output, n);
    }
    double silu_avx2_time = (get_time() - start) / iterations;
    printf("   AVX2:   %.4f ms (%.1fx speedup)\n\n", 
           silu_avx2_time * 1000, silu_scalar_time / silu_avx2_time);
    
    /* SwiGLU */
    printf("2. SwiGLU (gate * sigmoid(gate) * up):\n");
    
    /* Scalar */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        swiglu_scalar(input, input2, output, n);
    }
    double swiglu_scalar_time = (get_time() - start) / iterations;
    printf("   Scalar: %.4f ms\n", swiglu_scalar_time * 1000);
    
    /* AVX2 */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        swiglu_avx2(input, input2, output, n);
    }
    double swiglu_avx2_time = (get_time() - start) / iterations;
    printf("   AVX2:   %.4f ms (%.1fx speedup)\n\n",
           swiglu_avx2_time * 1000, swiglu_scalar_time / swiglu_avx2_time);
    
    /* RMSNorm */
    printf("3. RMSNorm (x / sqrt(mean(x^2) + eps)):\n");
    
    /* Scalar */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        rms_norm_scalar(input, output, n, 1e-5f);
    }
    double rms_scalar_time = (get_time() - start) / iterations;
    printf("   Scalar: %.4f ms\n", rms_scalar_time * 1000);
    
    /* AVX2 */
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        rms_norm_avx2(input, output, n, 1e-5f);
    }
    double rms_avx2_time = (get_time() - start) / iterations;
    printf("   AVX2:   %.4f ms (%.1fx speedup)\n\n",
           rms_avx2_time * 1000, rms_scalar_time / rms_avx2_time);
    
    /* Total impact on full model */
    double total_scalar = silu_scalar_time + swiglu_scalar_time + rms_scalar_time;
    double total_avx2 = silu_avx2_time + swiglu_avx2_time + rms_avx2_time;
    double total_speedup = total_scalar / total_avx2;
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Total activation time (scalar): %.3f ms                        ║\n", 
           total_scalar * 1000);
    printf("║  Total activation time (AVX2):   %.3f ms                        ║\n",
           total_avx2 * 1000);
    printf("║  Overall speedup:                %.1fx                          ║\n",
           total_speedup);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    
    /* Estimate impact on full model */
    double old_overhead_pct = 53.5;
    double new_overhead_pct = old_overhead_pct / total_speedup;
    double old_tok = 29.8;
    double new_tok = 1.0 / ((1.0 / old_tok) * (1.0 - old_overhead_pct/100 + new_overhead_pct/100));
    
    printf("║  Estimated impact:                                              ║\n");
    printf("║    Old overhead:  %.1f%%                                         ║\n", old_overhead_pct);
    printf("║    New overhead:  %.1f%%                                         ║\n", new_overhead_pct);
    printf("║    Old speed:     %.1f tok/sec                                  ║\n", old_tok);
    printf("║    New speed:     %.1f tok/sec                                  ║\n", new_tok);
    
    if (new_tok >= 50.0) {
        printf("║  ✅ Target achieved with AVX2 activations!                       ║\n");
    } else {
        printf("║  ⚠️  Need additional optimizations                               ║\n");
    }
    
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    aligned_free(input);
    aligned_free(input2);
    aligned_free(output);
    
    return 0;
}
