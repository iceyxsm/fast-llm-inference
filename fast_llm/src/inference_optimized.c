/*
 * Optimized Inference Engine with 6x16 kernel integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "model_loader.h"
#include "dequantized_tensor.h"

/* External optimized kernels */
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Optimized SwiGLU FFN using 6x16 kernel */
static void ffn_swiglu_optimized(const float* input, float* output,
                                  const dequantized_tensor_t* gate_proj,
                                  const dequantized_tensor_t* up_proj,
                                  const dequantized_tensor_t* down_proj,
                                  float* tmp_buffer,
                                  int hidden_size, int intermediate_size) {
    /* Gate projection: [intermediate, hidden] @ [hidden] -> [intermediate] */
    matmul_dequantized_asm_style(input, gate_proj, tmp_buffer, 1, intermediate_size, hidden_size);
    
    /* Apply SiLU to gate */
    for (int i = 0; i < intermediate_size; i++) {
        tmp_buffer[i] = silu(tmp_buffer[i]);
    }
    
    /* Up projection: [intermediate, hidden] @ [hidden] -> [intermediate] */
    /* Element-wise multiply with gate (SwiGLU) */
    float* up_out = tmp_buffer + intermediate_size;
    matmul_dequantized_asm_style(input, up_proj, up_out, 1, intermediate_size, hidden_size);
    
    for (int i = 0; i < intermediate_size; i++) {
        tmp_buffer[i] *= up_out[i];
    }
    
    /* Down projection: [hidden, intermediate] @ [intermediate] -> [hidden] */
    matmul_dequantized_asm_style(tmp_buffer, down_proj, output, 1, hidden_size, intermediate_size);
}

/* Quick RMS norm (no weight scaling for now) */
static void rms_norm_fast(const float* x, float* out, int n) {
    float sum_sq = 0.0f;
    #pragma omp simd reduction(+:sum_sq)
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + 1e-5f);
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

/* Optimized single layer */
static void transformer_layer_opt(transformer_model_t* model, int layer_idx,
                                   float* hidden_states, int seq_len) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    int num_heads = model->config.num_heads;
    int head_dim = model->config.head_dim;
    
    (void)num_heads; (void)head_dim; /* Attention not optimized yet */
    
    /* Allocate temporaries */
    float* residual = aligned_malloc(hidden_size * sizeof(float), 32);
    float* normed = aligned_malloc(hidden_size * sizeof(float), 32);
    float* ffn_tmp = aligned_malloc(2 * intermediate_size * sizeof(float), 32);
    float* ffn_out = aligned_malloc(hidden_size * sizeof(float), 32);
    
    /* Save residual */
    memcpy(residual, hidden_states, hidden_size * sizeof(float));
    
    /* Pre-FFN RMS norm */
    rms_norm_fast(hidden_states, normed, hidden_size);
    
    /* FFN with SwiGLU */
    if (model->gate_proj[layer_idx] && model->up_proj[layer_idx] && model->down_proj[layer_idx]) {
        ffn_swiglu_optimized(normed, ffn_out,
                             model->gate_proj[layer_idx],
                             model->up_proj[layer_idx],
                             model->down_proj[layer_idx],
                             ffn_tmp,
                             hidden_size, intermediate_size);
    }
    
    /* Residual connection */
    for (int i = 0; i < hidden_size; i++) {
        hidden_states[i] = residual[i] + ffn_out[i];
    }
    
    aligned_free(residual);
    aligned_free(normed);
    aligned_free(ffn_tmp);
    aligned_free(ffn_out);
}

/* Optimized forward pass - FFN only (for benchmark) */
double model_forward_ffn_only(transformer_model_t* model, int num_tokens) {
    int hidden_size = model->config.hidden_size;
    int num_layers = model->config.num_layers;
    
    /* Allocate hidden state */
    float* hidden = aligned_malloc(hidden_size * sizeof(float), 32);
    
    /* Random init */
    for (int i = 0; i < hidden_size; i++) {
        hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Running FFN-only benchmark: %d tokens, %d layers\n", num_tokens, num_layers);
    
    /* Warmup */
    for (int w = 0; w < 5; w++) {
        for (int layer = 0; layer < num_layers; layer++) {
            transformer_layer_opt(model, layer, hidden, 1);
        }
    }
    
    /* Benchmark */
    clock_t start = clock();
    
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < num_layers; layer++) {
            transformer_layer_opt(model, layer, hidden, 1);
        }
        
        /* Re-randomize hidden state for next token */
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tokens_per_sec = num_tokens / elapsed;
    
    printf("\n=== FFN-Only Benchmark ===\n");
    printf("Tokens: %d\n", num_tokens);
    printf("Layers: %d\n", num_layers);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tokens_per_sec);
    printf("==========================\n\n");
    
    aligned_free(hidden);
    
    return tokens_per_sec;
}
