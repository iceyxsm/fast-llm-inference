/*
 * Model Loader - GGUF format support
 * Simplified implementation for demonstration
 */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <stdint.h>
#include <stdbool.h>
#include "dequantized_tensor.h"
#include "speculative.h"
#include "medusa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model architecture configuration */
typedef struct {
    /* Architecture */
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    
    /* Quantization */
    int quant_bits;  /* 2, 4, or 8 */
    
    /* File info */
    char model_name[256];
    int file_version;
} model_config_t;

/* Complete model with all weights */
typedef struct {
    model_config_t config;
    
    /* Embeddings */
    float* token_embeddings;  /* [vocab_size, hidden_size] */
    
    /* Transformer layers - using pre-dequantized INT8 */
    dequantized_tensor_t** q_proj;  /* [num_layers] */
    dequantized_tensor_t** k_proj;
    dequantized_tensor_t** v_proj;
    dequantized_tensor_t** o_proj;
    dequantized_tensor_t** gate_proj;
    dequantized_tensor_t** up_proj;
    dequantized_tensor_t** down_proj;
    
    /* Layer norms */
    float** input_layernorm;
    float** post_attn_layernorm;
    
    /* LM head */
    dequantized_tensor_t* lm_head;
    
    /* Optional: Speculative draft model */
    draft_model_t* draft_model;
    
    /* Optional: Medusa heads */
    medusa_model_t* medusa_model;
    
    /* KV cache */
    float* k_cache;
    float* v_cache;
    int cache_pos;
} transformer_model_t;

/* Load model from GGUF file */
transformer_model_t* model_load_gguf(const char* path, int use_int8);

/* Create mock model for testing/benchmarking */
transformer_model_t* model_create_mock(int hidden_size, int intermediate_size, 
                                        int num_layers, int vocab_size);

/* Free model */
void model_free(transformer_model_t* model);

/* Print model info */
void model_print_info(const transformer_model_t* model);

/* 
 * Run inference - single forward pass
 * input_tokens: [seq_len]
 * output_logits: [seq_len, vocab_size] (only last position used for generation)
 */
void model_forward(transformer_model_t* model, 
                   const int* input_tokens, int seq_len,
                   float* output_logits, int* output_tokens);

/* Generate tokens autoregressively */
int model_generate(transformer_model_t* model,
                   const char* prompt,
                   int* output_tokens,
                   int max_tokens,
                   float temperature,
                   bool use_speculative,
                   bool use_medusa);

/* Benchmark model speed */
double model_benchmark(transformer_model_t* model, int num_tokens, bool use_optimizations);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_LOADER_H */
