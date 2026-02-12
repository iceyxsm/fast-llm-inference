/*
 * Benchmark optimized model inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include "cpu_features.h"

extern double model_forward_ffn_only(transformer_model_t* model, int num_tokens);
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);

/* Create a synthetic model for testing */
transformer_model_t* create_test_model(int num_layers, int hidden_size, int intermediate_size) {
    transformer_model_t* model = calloc(1, sizeof(transformer_model_t));
    
    model->config.num_layers = num_layers;
    model->config.hidden_size = hidden_size;
    model->config.intermediate_size = intermediate_size;
    model->config.num_heads = 32;
    model->config.head_dim = hidden_size / 32;
    
    /* Allocate projection arrays */
    model->gate_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->up_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->down_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    
    /* Create random weights for each layer */
    for (int l = 0; l < num_layers; l++) {
        /* Gate: [intermediate, hidden] */
        model->gate_proj[l] = calloc(1, sizeof(dequantized_tensor_t));
        model->gate_proj[l]->rows = intermediate_size;
        model->gate_proj[l]->cols = hidden_size;
        model->gate_proj[l]->weights = aligned_malloc(intermediate_size * hidden_size, 64);
        model->gate_proj[l]->scales = aligned_malloc(intermediate_size * sizeof(float), 64);
        
        for (int r = 0; r < intermediate_size; r++) {
            model->gate_proj[l]->scales[r] = 0.01f;
            for (int c = 0; c < hidden_size; c++) {
                model->gate_proj[l]->weights[r * hidden_size + c] = (rand() % 256) - 128;
            }
        }
        
        /* Up: [intermediate, hidden] */
        model->up_proj[l] = calloc(1, sizeof(dequantized_tensor_t));
        model->up_proj[l]->rows = intermediate_size;
        model->up_proj[l]->cols = hidden_size;
        model->up_proj[l]->weights = aligned_malloc(intermediate_size * hidden_size, 64);
        model->up_proj[l]->scales = aligned_malloc(intermediate_size * sizeof(float), 64);
        
        for (int r = 0; r < intermediate_size; r++) {
            model->up_proj[l]->scales[r] = 0.01f;
            for (int c = 0; c < hidden_size; c++) {
                model->up_proj[l]->weights[r * hidden_size + c] = (rand() % 256) - 128;
            }
        }
        
        /* Down: [hidden, intermediate] */
        model->down_proj[l] = calloc(1, sizeof(dequantized_tensor_t));
        model->down_proj[l]->rows = hidden_size;
        model->down_proj[l]->cols = intermediate_size;
        model->down_proj[l]->weights = aligned_malloc(hidden_size * intermediate_size, 64);
        model->down_proj[l]->scales = aligned_malloc(hidden_size * sizeof(float), 64);
        
        for (int r = 0; r < hidden_size; r++) {
            model->down_proj[l]->scales[r] = 0.01f;
            for (int c = 0; c < intermediate_size; c++) {
                model->down_proj[l]->weights[r * intermediate_size + c] = (rand() % 256) - 128;
            }
        }
    }
    
    return model;
}

void free_test_model(transformer_model_t* model) {
    for (int l = 0; l < model->config.num_layers; l++) {
        if (model->gate_proj[l]) {
            aligned_free(model->gate_proj[l]->weights);
            aligned_free(model->gate_proj[l]->scales);
            free(model->gate_proj[l]);
        }
        if (model->up_proj[l]) {
            aligned_free(model->up_proj[l]->weights);
            aligned_free(model->up_proj[l]->scales);
            free(model->up_proj[l]);
        }
        if (model->down_proj[l]) {
            aligned_free(model->down_proj[l]->weights);
            aligned_free(model->down_proj[l]->scales);
            free(model->down_proj[l]);
        }
    }
    free(model->gate_proj);
    free(model->up_proj);
    free(model->down_proj);
    free(model);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  OPTIMIZED MODEL INFERENCE BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    /* Phi-3 mini config */
    int num_layers = 32;
    int hidden_size = 3072;
    int intermediate_size = 8192;
    int num_tokens = 100;
    
    /* Allow command-line override */
    if (argc > 1) num_tokens = atoi(argv[1]);
    
    printf("Model config: %d layers, %d hidden, %d intermediate\n",
           num_layers, hidden_size, intermediate_size);
    printf("Benchmark: %d tokens\n\n", num_tokens);
    
    /* Create test model */
    printf("Creating test model...\n");
    transformer_model_t* model = create_test_model(num_layers, hidden_size, intermediate_size);
    
    /* Run benchmark */
    double tok_per_sec = model_forward_ffn_only(model, num_tokens);
    
    /* Summary */
    printf("\n=== SUMMARY ===\n");
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("Target: 50 tok/sec\n");
    printf("Gap: %.1f%%\n", (tok_per_sec / 50.0) * 100.0);
    
    if (tok_per_sec >= 50.0) {
        printf("\n🎉🎉🎉 50 TOK/SEC ACHIEVED! 🎉🎉🎉\n");
    } else if (tok_per_sec >= 45.0) {
        printf("\n⚠️ Very close! (90%%+)\n");
    } else {
        printf("\nNeed %.2fx speedup to reach 50 tok/sec\n", 50.0 / tok_per_sec);
    }
    
    /* Cleanup */
    free_test_model(model);
    
    return 0;
}
