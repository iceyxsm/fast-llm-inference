/*
 * Overhead Analysis: Why is full model slower than matmul alone?
 * 
 * Isolated matmul: 60 tok/sec
 * Full model: 30 tok/sec
 * Overhead: 50% of time is NOT in matmul!
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

/* Already declared in header */

/* Minimal RMS norm */
static void rms_norm(const float* x, float* out, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

/* SiLU activation */
static void silu(const float* x, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* Element-wise multiply */
static void mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

/* Add residual */
static void add(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  OVERHEAD ANALYSIS: Where is the time going?                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    int hidden = HIDDEN_SIZE;
    int intermediate = INTERMEDIATE_SIZE;
    int iterations = 100;
    
    /* Setup weights */
    dequantized_tensor_t gate_proj, up_proj, down_proj;
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
        gate_proj.scales[i] = 0.01f;
        up_proj.scales[i] = 0.01f;
    }
    for (int i = 0; i < hidden; i++) {
        down_proj.scales[i] = 0.01f;
    }
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 32);
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Pre-allocate all buffers */
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* gate_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* swiglu_out = (float*)aligned_malloc(intermediate * sizeof(float), 32);
    float* down_out = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    printf("Benchmarking one FFN layer (%d iterations)...\n\n", iterations);
    
    /* Test 1: Matmul only (3 matmuls) */
    printf("1. MATMUL ONLY (3 matmuls):\n");
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        matmul_dequantized_asm_style(hidden_state, &gate_proj, gate_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(hidden_state, &up_proj, up_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(swiglu_out, &down_proj, down_out, 1, hidden, intermediate);
    }
    double matmul_time = (get_time() - start) / iterations;
    printf("   Time: %.3f ms\n\n", matmul_time * 1000);
    
    /* Test 2: Matmul + activations */
    printf("2. MATMUL + ACTIVATIONS:\n");
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        matmul_dequantized_asm_style(hidden_state, &gate_proj, gate_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(hidden_state, &up_proj, up_out, 1, intermediate, hidden);
        /* SwiGLU */
        for (int i = 0; i < intermediate; i++) {
            swiglu_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
        }
        matmul_dequantized_asm_style(swiglu_out, &down_proj, down_out, 1, hidden, intermediate);
    }
    double with_activation_time = (get_time() - start) / iterations;
    printf("   Time: %.3f ms (activation overhead: %.1f%%)\n\n", 
           with_activation_time * 1000, 
           (with_activation_time - matmul_time) / with_activation_time * 100);
    
    /* Test 3: Full FFN (matmul + activations + norms + residuals) */
    printf("3. FULL FFN (all operations):\n");
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        /* RMS Norm */
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) sum_sq += hidden_state[i] * hidden_state[i];
        float scale = 1.0f / sqrtf(sum_sq / hidden + 1e-5f);
        for (int i = 0; i < hidden; i++) norm_out[i] = hidden_state[i] * scale;
        
        /* Gate */
        matmul_dequantized_asm_style(norm_out, &gate_proj, gate_out, 1, intermediate, hidden);
        /* Up */
        matmul_dequantized_asm_style(norm_out, &up_proj, up_out, 1, intermediate, hidden);
        /* SwiGLU */
        for (int i = 0; i < intermediate; i++) {
            swiglu_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
        }
        /* Down */
        matmul_dequantized_asm_style(swiglu_out, &down_proj, down_out, 1, hidden, intermediate);
        /* Residual */
        for (int i = 0; i < hidden; i++) {
            hidden_state[i] = hidden_state[i] + down_out[i];
        }
    }
    double full_ffn_time = (get_time() - start) / iterations;
    printf("   Time: %.3f ms\n\n", full_ffn_time * 1000);
    
    /* Test 4: Full model simulation (32 layers) */
    printf("4. FULL MODEL (32 layers):\n");
    start = get_time();
    for (int iter = 0; iter < 20; iter++) {
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            /* Norm */
            float sum_sq = 0.0f;
            for (int i = 0; i < hidden; i++) sum_sq += hidden_state[i] * hidden_state[i];
            float scale = 1.0f / sqrtf(sum_sq / hidden + 1e-5f);
            for (int i = 0; i < hidden; i++) norm_out[i] = hidden_state[i] * scale;
            
            /* Gate */
            matmul_dequantized_asm_style(norm_out, &gate_proj, gate_out, 1, intermediate, hidden);
            /* Up */
            matmul_dequantized_asm_style(norm_out, &up_proj, up_out, 1, intermediate, hidden);
            /* SwiGLU */
            for (int i = 0; i < intermediate; i++) {
                swiglu_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
            }
            /* Down */
            matmul_dequantized_asm_style(swiglu_out, &down_proj, down_out, 1, hidden, intermediate);
            /* Residual */
            for (int i = 0; i < hidden; i++) {
                hidden_state[i] = hidden_state[i] + down_out[i];
            }
        }
    }
    double full_model_time = (get_time() - start) / 20;
    double full_model_tok = 1.0 / full_model_time;
    printf("   Time per token: %.3f ms\n", full_model_time * 1000);
    printf("   Tok/sec: %.1f\n\n", full_model_tok);
    
    /* Analysis */
    double matmul_pct = matmul_time / full_ffn_time * 100;
    double overhead_pct = 100 - matmul_pct;
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  ANALYSIS                                                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Matmul time (3 matmuls): %.3f ms (%.0f%% of FFN)               ║\n", 
           matmul_time * 1000, matmul_pct);
    printf("║  Other operations:        %.3f ms (%.0f%% of FFN)               ║\n",
           (full_ffn_time - matmul_time) * 1000, overhead_pct);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Theoretical max (matmul only): %.1f tok/sec                    ║\n",
           1.0 / (matmul_time * NUM_LAYERS));
    printf("║  Actual full model:            %.1f tok/sec                     ║\n",
           full_model_tok);
    printf("║  Overhead penalty:             %.1f%%                            ║\n",
           (1.0 / (matmul_time * NUM_LAYERS) - full_model_tok) / (1.0 / (matmul_time * NUM_LAYERS)) * 100);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
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
    aligned_free(swiglu_out);
    aligned_free(down_out);
    
    return 0;
}
