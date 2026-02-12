/*
 * Benchmark with Real GGUF Weights
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
#include "cpu_features.h"

/* Simple layer forward using real loaded weights */
void layer_forward_real(transformer_model_t* model, int layer_idx,
                        float* input, float* output, int seq_len) {
    int hidden = model->config.hidden_size;
    int intermediate = model->config.intermediate_size;
    
    /* Temp buffers */
    float* gate = aligned_malloc(seq_len * intermediate * sizeof(float), 32);
    float* up = aligned_malloc(seq_len * intermediate * sizeof(float), 32);
    
    /* FFN Gate projection */
    if (model->gate_proj[layer_idx]) {
        matmul_dequantized(input, model->gate_proj[layer_idx], gate, seq_len, intermediate, hidden);
    }
    
    /* FFN Up projection */
    if (model->up_proj[layer_idx]) {
        matmul_dequantized(input, model->up_proj[layer_idx], up, seq_len, intermediate, hidden);
    }
    
    /* SiLU + Multiply */
    for (int i = 0; i < seq_len * intermediate; i++) {
        float g = gate[i];
        float sigmoid = 1.0f / (1.0f + expf(-g));
        gate[i] = g * sigmoid * up[i];
    }
    
    /* FFN Down projection */
    if (model->down_proj[layer_idx]) {
        matmul_dequantized(gate, model->down_proj[layer_idx], output, seq_len, hidden, intermediate);
    }
    
    /* Residual */
    for (int i = 0; i < seq_len * hidden; i++) {
        output[i] += input[i];
    }
    
    aligned_free(gate);
    aligned_free(up);
}

/* Benchmark with real model */
void benchmark_real_model(const char* model_path, int num_tokens) {
    printf("\n========================================\n");
    printf("  BENCHMARK WITH REAL GGUF WEIGHTS\n");
    printf("========================================\n\n");
    
    /* Load real model */
    printf("Loading model from: %s\n", model_path);
    transformer_model_t* model = model_load_gguf(model_path, 1);
    
    if (!model) {
        printf("Failed to load real model, falling back to mock\n");
        model = model_create_mock(3072, 8192, 32, 32064);
    }
    
    model_print_info(model);
    
    int hidden = model->config.hidden_size;
    int num_layers = model->config.num_layers;
    
    /* Allocate buffers */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output = aligned_malloc(hidden * sizeof(float), 32);
    
    /* Random input */
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int w = 0; w < 3; w++) {
        for (int l = 0; l < num_layers; l++) {
            layer_forward_real(model, l, input, output, 1);
            float* tmp = input; input = output; output = tmp;
        }
    }
    
    /* Benchmark */
    printf("\nBenchmarking %d tokens...\n", num_tokens);
    clock_t start = clock();
    
    for (int t = 0; t < num_tokens; t++) {
        for (int l = 0; l < num_layers; l++) {
            layer_forward_real(model, l, input, output, 1);
            float* tmp = input; input = output; output = tmp;
        }
        
        if ((t + 1) % 10 == 0) {
            printf("  Generated %d/%d tokens...\r", t + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tokens_per_sec = num_tokens / elapsed;
    
    printf("\n\n=== RESULTS ===\n");
    printf("Model: %s\n", model->config.model_name);
    printf("Tokens: %d\n", num_tokens);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tokens_per_sec);
    printf("Ms/token: %.2f ms\n", 1000.0 / tokens_per_sec);
    printf("\n");
    
    /* Cleanup */
    aligned_free(input);
    aligned_free(output);
    model_free(model);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    const char* model_path = "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    int num_tokens = 50;
    
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_tokens = atoi(argv[++i]);
        }
    }
    
    cpu_features_t cpu = detect_cpu_features();
    printf("\nCPU: AVX2=%s, Cores=%d\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    benchmark_real_model(model_path, num_tokens);
    
    printf("\nDone!\n\n");
    return 0;
}
