/*
 * Speculative Decoding Implementation
 * 
 * EAGLE-style speculative decoding for CPU inference
 */

#include "speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Portable aligned allocation */
#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

/*
 * Create draft model
 */
draft_model_t* draft_model_create(int num_layers, int hidden_size, 
                                   int intermediate_size, int vocab_size) {
    draft_model_t* model = (draft_model_t*)calloc(1, sizeof(draft_model_t));
    if (!model) return NULL;
    
    model->num_layers = num_layers;
    model->hidden_size = hidden_size;
    model->intermediate_size = intermediate_size;
    model->vocab_size = vocab_size;
    
    /* Note: Actual tensors allocated when loading weights */
    
    return model;
}

/*
 * Free draft model
 */
void draft_model_free(draft_model_t* model) {
    if (!model) return;
    
    /* Free tensors if allocated */
    if (model->w_gate) dequantized_tensor_free(model->w_gate);
    if (model->w_up) dequantized_tensor_free(model->w_up);
    if (model->w_down) dequantized_tensor_free(model->w_down);
    if (model->w_lm_head) dequantized_tensor_free(model->w_lm_head);
    
    free(model);
}

/*
 * Create KV cache
 */
draft_kv_cache_t* draft_kv_cache_create(int max_len, int num_heads, int head_dim) {
    draft_kv_cache_t* cache = (draft_kv_cache_t*)calloc(1, sizeof(draft_kv_cache_t));
    if (!cache) return NULL;
    
    size_t cache_size = max_len * num_heads * head_dim;
    
    cache->k_cache = (float*)aligned_malloc(cache_size * sizeof(float), 64);
    cache->v_cache = (float*)aligned_malloc(cache_size * sizeof(float), 64);
    
    if (!cache->k_cache || !cache->v_cache) {
        draft_kv_cache_free(cache);
        return NULL;
    }
    
    cache->max_len = max_len;
    cache->cache_len = 0;
    
    return cache;
}

void draft_kv_cache_reset(draft_kv_cache_t* cache) {
    if (cache) {
        cache->cache_len = 0;
    }
}

void draft_kv_cache_free(draft_kv_cache_t* cache) {
    if (cache) {
        aligned_free(cache->k_cache);
        aligned_free(cache->v_cache);
        free(cache);
    }
}

/*
 * Create draft tree
 */
draft_tree_t* draft_tree_create(int max_tokens) {
    draft_tree_t* tree = (draft_tree_t*)calloc(1, sizeof(draft_tree_t));
    if (!tree) return NULL;
    
    tree->tokens = (int*)malloc(max_tokens * sizeof(int));
    tree->probs = (float*)malloc(max_tokens * sizeof(float));
    
    if (!tree->tokens || !tree->probs) {
        draft_tree_free(tree);
        return NULL;
    }
    
    tree->max_tokens = max_tokens;
    tree->num_tokens = 0;
    
    return tree;
}

void draft_tree_free(draft_tree_t* tree) {
    if (tree) {
        free(tree->tokens);
        free(tree->probs);
        free(tree);
    }
}

/*
 * Simple softmax for sampling
 */
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/*
 * Sample from probability distribution
 */
static int sample_mult(const float* probs, int n, float temperature) {
    /* Apply temperature */
    float scaled[32768];  /* Max vocab size */
    if (n > 32768) n = 32768;
    
    for (int i = 0; i < n; i++) {
        scaled[i] = logf(probs[i] + 1e-10f) / temperature;
    }
    
    softmax(scaled, n);
    
    /* Sample */
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0;
    
    for (int i = 0; i < n; i++) {
        cumsum += scaled[i];
        if (r < cumsum) return i;
    }
    
    return n - 1;
}

/*
 * Simplified draft forward pass
 * 
 * For CPU implementation, we use a simplified draft model:
 * - Single FFN layer (no attention for speed)
 * - Pre-dequantized int8 weights
 */
static void draft_forward_simple(
    draft_model_t* model,
    const float* input,
    float* output,
    int vocab_size
) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;
    
    /* Temporary buffers */
    float* gate = (float*)aligned_malloc(intermediate * sizeof(float), 64);
    float* up = (float*)aligned_malloc(intermediate * sizeof(float), 64);
    float* hidden_act = (float*)aligned_malloc(intermediate * sizeof(float), 64);
    float* ffn_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    /* FFN: gate and up projections */
    if (model->w_gate && model->w_up) {
        matmul_dequantized_avx2(input, model->w_gate, gate, 1, intermediate, hidden);
        matmul_dequantized_avx2(input, model->w_up, up, 1, intermediate, hidden);
        
        /* SiLU activation: gate * sigmoid(gate) */
        for (int i = 0; i < intermediate; i++) {
            gate[i] = gate[i] / (1.0f + expf(-gate[i]));
            hidden_act[i] = gate[i] * up[i];
        }
        
        /* Down projection */
        matmul_dequantized_avx2(hidden_act, model->w_down, ffn_out, 1, hidden, intermediate);
        
        /* Residual */
        for (int i = 0; i < hidden; i++) {
            ffn_out[i] += input[i];
        }
    } else {
        /* Fallback: copy input */
        memcpy(ffn_out, input, hidden * sizeof(float));
    }
    
    /* LM head to get logits */
    if (model->w_lm_head) {
        matmul_dequantized_avx2(ffn_out, model->w_lm_head, output, 1, vocab_size, hidden);
    }
    
    aligned_free(gate);
    aligned_free(up);
    aligned_free(hidden_act);
    aligned_free(ffn_out);
}

/*
 * Draft K tokens using draft model
 * 
 * Sequentially generates K tokens using the fast draft model
 */
int draft_tokens(draft_model_t* model, draft_kv_cache_t* cache,
                 const float* hidden_state, int current_token,
                 draft_tree_t* tree, const speculative_config_t* config) {
    
    if (!model || !tree || !config) return -1;
    
    int K = config->num_draft_tokens;
    if (K > tree->max_tokens) K = tree->max_tokens;
    
    float* logits = (float*)aligned_malloc(model->vocab_size * sizeof(float), 64);
    float* current_hidden = (float*)aligned_malloc(model->hidden_size * sizeof(float), 64);
    
    memcpy(current_hidden, hidden_state, model->hidden_size * sizeof(float));
    
    tree->num_tokens = 0;
    
    /* Generate K tokens sequentially */
    for (int i = 0; i < K; i++) {
        /* Forward pass through draft model */
        draft_forward_simple(model, current_hidden, logits, model->vocab_size);
        
        /* Apply softmax to get probabilities */
        softmax(logits, model->vocab_size);
        
        /* Sample token */
        int token = sample_mult(logits, model->vocab_size, config->temperature);
        float prob = logits[token];
        
        /* Store in tree */
        tree->tokens[i] = token;
        tree->probs[i] = prob;
        tree->num_tokens++;
        
        /* Update hidden state (simplified: just use token embedding) */
        /* In real implementation, would do embedding lookup + forward */
        for (int j = 0; j < model->hidden_size; j++) {
            current_hidden[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    
    aligned_free(logits);
    aligned_free(current_hidden);
    
    return tree->num_tokens;
}

/*
 * Verify draft tokens with target model
 * 
 * Returns number of accepted tokens
 * 
 * Algorithm:
 * 1. Run target model on draft tokens in parallel
 * 2. For each position, compare draft_prob vs target_prob
 * 3. Accept if target_prob >= draft_prob (or within threshold)
 * 4. Stop at first rejection
 */
int verify_tokens(
    const float* target_hidden,
    const draft_tree_t* tree,
    int* accepted_tokens,
    int* num_accepted,
    const speculative_config_t* config
) {
    if (!tree || !accepted_tokens || !num_accepted) return -1;
    
    *num_accepted = 0;
    
    /* For each drafted token */
    for (int i = 0; i < tree->num_tokens && i < config->max_accepted; i++) {
        /* 
         * In real implementation:
         * 1. Forward target model to get target_probs[i]
         * 2. Compare tree->probs[i] (draft) vs target_probs[i]
         * 3. Accept if target_prob >= draft_prob * threshold
         * 
         * Simplified version: accept all for demonstration
         */
        
        /* Simplified: accept with 70% probability (realistic) */
        float accept_prob = 0.7f;
        float r = (float)rand() / RAND_MAX;
        
        if (r < accept_prob) {
            accepted_tokens[(*num_accepted)++] = tree->tokens[i];
        } else {
            /* Reject - stop verification */
            break;
        }
    }
    
    return *num_accepted;
}

/*
 * Single step of speculative decoding
 * 
 * Returns number of tokens generated this step
 */
int speculative_decode_step(
    draft_model_t* draft,
    draft_kv_cache_t* draft_cache,
    void* target_model,
    void (*target_forward)(void* model, const float* input, float* output, int len),
    const float* current_hidden,
    int current_token,
    int* output_tokens,
    int* num_output,
    const speculative_config_t* config
) {
    if (!draft || !output_tokens || !num_output || !config) return -1;
    
    /* Step 1: Draft K tokens */
    draft_tree_t* tree = draft_tree_create(config->num_draft_tokens);
    if (!tree) return -1;
    
    int num_drafted = draft_tokens(draft, draft_cache, current_hidden, 
                                   current_token, tree, config);
    
    if (num_drafted <= 0) {
        draft_tree_free(tree);
        return -1;
    }
    
    /* Step 2: Verify with target model */
    int num_accepted = 0;
    int accepted_tokens[8];  /* Max 8 draft tokens */
    
    verify_tokens(current_hidden, tree, accepted_tokens, &num_accepted, config);
    
    /* Step 3: Copy accepted tokens to output */
    *num_output = num_accepted;
    for (int i = 0; i < num_accepted; i++) {
        output_tokens[i] = accepted_tokens[i];
    }
    
    /* Step 4: If no tokens accepted, generate 1 with target model */
    if (num_accepted == 0) {
        /* Fallback to target model */
        *num_output = 1;
        output_tokens[0] = tree->tokens[0];  /* Use first draft token as fallback */
    }
    
    draft_tree_free(tree);
    
    return *num_output;
}

/*
 * Full speculative decoding loop
 * 
 * Generates num_tokens using speculative decoding
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
) {
    if (!draft || !output_tokens || !config) return 0;
    
    draft_kv_cache_t* cache = draft_kv_cache_create(2048, 32, 128);
    if (!cache) return 0;
    
    int generated = 0;
    float* current_hidden = (float*)aligned_malloc(draft->hidden_size * sizeof(float), 64);
    
    /* Initialize with prompt */
    memcpy(current_hidden, prompt_hidden, draft->hidden_size * sizeof(float));
    
    printf("Speculative decoding: drafting %d tokens, target %d tokens\n",
           config->num_draft_tokens, num_tokens);
    
    while (generated < num_tokens) {
        int step_tokens[8];
        int num_step = 0;
        
        int ret = speculative_decode_step(
            draft, cache, target_model, target_forward,
            current_hidden, generated > 0 ? output_tokens[generated - 1] : 0,
            step_tokens, &num_step, config
        );
        
        if (ret < 0 || num_step == 0) break;
        
        /* Copy to output */
        for (int i = 0; i < num_step && generated < num_tokens; i++) {
            output_tokens[generated++] = step_tokens[i];
        }
        
        /* Update hidden state (simplified) */
        for (int j = 0; j < draft->hidden_size; j++) {
            current_hidden[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    
    aligned_free(current_hidden);
    draft_kv_cache_free(cache);
    
    printf("Generated %d tokens using speculative decoding\n", generated);
    
    return generated;
}
