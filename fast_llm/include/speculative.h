/*
 * Speculative Decoding - EAGLE-style Implementation
 * 
 * Research-based implementation:
 * - EAGLE-3 paper: https://arxiv.org/abs/2503.01840
 * - Draft model generates K tokens cheaply
 * - Target model verifies in parallel with tree attention
 * - Accept tokens that match target distribution
 * 
 * Expected speedup: 2-3x (conservative for CPU)
 */

#ifndef SPECULATIVE_H
#define SPECULATIVE_H

#include <stdint.h>
#include <stdbool.h>
#include "dequantized_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Speculative decoding configuration */
typedef struct {
    int num_draft_tokens;      /* K: number of tokens to draft (typically 4-8) */
    float temperature;         /* Sampling temperature */
    int max_accepted;          /* Max tokens to accept per step */
} speculative_config_t;

/* Draft model - smaller version of target model */
typedef struct {
    /* Simplified transformer layers (4-8 layers vs 32) */
    dequantized_tensor_t* w_q_proj;      /* Query projection */
    dequantized_tensor_t* w_k_proj;      /* Key projection */
    dequantized_tensor_t* w_v_proj;      /* Value projection */
    dequantized_tensor_t* w_o_proj;      /* Output projection */
    
    dequantized_tensor_t* w_gate;        /* FFN gate */
    dequantized_tensor_t* w_up;          /* FFN up */
    dequantized_tensor_t* w_down;        /* FFN down */
    
    dequantized_tensor_t* w_lm_head;     /* LM head for token prediction */
    
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
} draft_model_t;

/* KV cache for draft model */
typedef struct {
    float* k_cache;    /* [max_seq_len, num_heads, head_dim] */
    float* v_cache;    /* [max_seq_len, num_heads, head_dim] */
    int cache_len;
    int max_len;
} draft_kv_cache_t;

/* Tree structure for draft tokens */
typedef struct {
    int* tokens;           /* Drafted token IDs */
    float* probs;          /* Draft model probabilities */
    int num_tokens;
    int max_tokens;
} draft_tree_t;

/* Default configuration */
static inline speculative_config_t speculative_default_config(void) {
    speculative_config_t cfg = {
        .num_draft_tokens = 4,    /* Draft 4 tokens */
        .temperature = 0.8f,
        .max_accepted = 4
    };
    return cfg;
}

/* Create draft model (simplified, fewer layers) */
draft_model_t* draft_model_create(int num_layers, int hidden_size, 
                                   int intermediate_size, int vocab_size);

/* Load draft model weights from pre-trained */
int draft_model_load(draft_model_t* model, const char* path);

/* Free draft model */
void draft_model_free(draft_model_t* model);

/* Create KV cache */
draft_kv_cache_t* draft_kv_cache_create(int max_len, int num_heads, int head_dim);

/* Reset KV cache */
void draft_kv_cache_reset(draft_kv_cache_t* cache);

/* Free KV cache */
void draft_kv_cache_free(draft_kv_cache_t* cache);

/* Create draft tree */
draft_tree_t* draft_tree_create(int max_tokens);

/* Free draft tree */
void draft_tree_free(draft_tree_t* tree);

/*
 * Draft K tokens using draft model
 * 
 * Input: current hidden state [hidden_size]
 * Output: tree with K drafted tokens and their probabilities
 */
int draft_tokens(draft_model_t* model, draft_kv_cache_t* cache,
                 const float* hidden_state, int current_token,
                 draft_tree_t* tree, const speculative_config_t* config);

/*
 * Verify draft tokens with target model
 * 
 * Input: draft tree with K tokens
 * Output: number of accepted tokens (0 to K)
 * 
 * Uses target model to compute probabilities for all K positions
 * Accepts tokens where draft_prob ≈ target_prob
 */
int verify_tokens(
    /* Target model (full) */
    const float* target_hidden,
    
    /* Draft tree */
    const draft_tree_t* tree,
    
    /* Output: accepted tokens */
    int* accepted_tokens,
    int* num_accepted,
    
    /* Sampling config */
    const speculative_config_t* config
);

/*
 * Single step of speculative decoding
 * 
 * 1. Draft K tokens with draft model (fast)
 * 2. Verify K tokens with target model (parallel)
 * 3. Accept M tokens (M <= K)
 * 4. Return accepted tokens and M
 * 
 * Speedup: ~K/M times faster than autoregressive
 */
int speculative_decode_step(
    /* Draft model */
    draft_model_t* draft,
    draft_kv_cache_t* draft_cache,
    
    /* Target model (function pointer for flexibility) */
    void* target_model,
    void (*target_forward)(void* model, const float* input, float* output, int len),
    
    /* Current state */
    const float* current_hidden,
    int current_token,
    
    /* Output */
    int* output_tokens,
    int* num_output,
    
    /* Config */
    const speculative_config_t* config
);

/*
 * Full speculative decoding loop
 * 
 * Generates num_tokens using speculative decoding
 * Returns actual tokens generated
 */
int speculative_generate(
    draft_model_t* draft,
    void* target_model,
    void (*target_forward)(void*, const float*, float*, int),
    
    const float* prompt_hidden,
    int prompt_len,
    
    int* output_tokens,
    int num_tokens,
    
    const speculative_config_t* config
);

#ifdef __cplusplus
}
#endif

#endif /* SPECULATIVE_H */
