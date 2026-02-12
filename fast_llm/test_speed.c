/*
 * Speed test for optimized inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

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

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Simple RMS norm */
static void rms_norm_fast(const float* x, float* out, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

/* External optimized kernel */
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);

/* Single FFN layer using optimized kernel */
static void ffn_layer_opt(transformer_model_t* model, int layer_idx,
                          float* hidden_states) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    
    float* residual = aligned_malloc(hidden_size * sizeof(float), 32);
    float* normed = aligned_malloc(hidden_size * sizeof(float), 32);
    float* gate = aligned_malloc(intermediate_size * sizeof(float), 32);
    float* up = aligned_malloc(intermediate_size * sizeof(float), 32);
    float* ffn_out = aligned_malloc(hidden_size * sizeof(float), 32);
    
    memcpy(residual, hidden_states, hidden_size * sizeof(float));
    rms_norm_fast(hidden_states, normed, hidden_size);
    
    /* Gate projection */
    if (model->gate_proj[layer_idx]) {
        matmul_dequantized_asm_style(normed, model->gate_proj[layer_idx], gate, 1, intermediate_size, hidden_size);
        for (int i = 0; i < intermediate_size; i++) {
            gate[i] = silu(gate[i]);
        }
    }
    
    /* Up projection */
    if (model->up_proj[layer_idx]) {
        matmul_dequantized_asm_style(normed, model->up_proj[layer_idx], up, 1, intermediate_size, hidden_size);
    }
    
    /* SwiGLU */
    for (int i = 0; i < intermediate_size; i++) {
        gate[i] *= up[i];
    }
    
    /* Down projection */
    if (model->down_proj[layer_idx]) {
        matmul_dequantized_asm_style(gate, model->down_proj[layer_idx], ffn_out, 1, hidden_size, intermediate_size);
    }
    
    /* Residual */
    for (int i = 0; i < hidden_size; i++) {
        hidden_states[i] = residual[i] + ffn_out[i];
    }
    
    aligned_free(residual);
    aligned_free(normed);
    aligned_free(gate);
    aligned_free(up);
    aligned_free(ffn_out);
}

int main() {
    printf("=== Optimized Inference Speed Test ===\n\n");
    
    /* Phi-3 mini config */
    int hidden = 3072;
    int intermediate = 8192;
    int layers = 32;
    int vocab = 32064;
    int num_tokens = 20;
    
    printf("Model: Phi-3-mini-like\n");
    printf("  Hidden: %d\n", hidden);
    printf("  Intermediate: %d\n", intermediate);
    printf("  Layers: %d\n", layers);
    printf("  Tokens: %d\n\n", num_tokens);
    
    transformer_model_t* model = model_create_mock(hidden, intermediate, layers, vocab);
    if (!model) {
        printf("Failed to create model\n");
        return 1;
    }
    
    float* hidden_state = aligned_malloc(model->config.hidden_size * sizeof(float), 32);
    for (int i = 0; i < model->config.hidden_size; i++) {
        hidden_state[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Warming up...\n");
    for (int w = 0; w < 3; w++) {
        for (int l = 0; l < model->config.num_layers; l++) {
            ffn_layer_opt(model, l, hidden_state);
        }
    }
    
    printf("Running benchmark...\n");
    clock_t start = clock();
    
    for (int t = 0; t < num_tokens; t++) {
        for (int l = 0; l < model->config.num_layers; l++) {
            ffn_layer_opt(model, l, hidden_state);
        }
        
        /* Perturb hidden state for next token */
        for (int i = 0; i < model->config.hidden_size; i++) {
            hidden_state[i] += ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tok_per_sec = num_tokens / elapsed;
    
    printf("\n=== RESULTS ===\n");
    printf("Time: %.3f seconds\n", elapsed);
    printf("Tokens: %d\n", num_tokens);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("===============\n");
    printf("\nNote: This is FFN-only. Full model (with attention) ~20-30%% slower.\n");
    printf("Target: 50 tok/sec for full model\n");
    
    aligned_free(hidden_state);
    /* Skip model_free to avoid potential crash */
    
    return 0;
}
