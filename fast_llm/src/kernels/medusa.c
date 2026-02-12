/*
 * Medusa Multi-Token Prediction Implementation
 * 
 * Adds extra heads to base model for parallel token prediction
 */

#include "medusa.h"
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
 * Create Medusa model with N heads
 */
medusa_model_t* medusa_model_create(int num_heads, int hidden_size, int vocab_size) {
    medusa_model_t* model = (medusa_model_t*)calloc(1, sizeof(medusa_model_t));
    if (!model) return NULL;
    
    model->num_heads = num_heads;
    model->hidden_size = hidden_size;
    model->vocab_size = vocab_size;
    
    model->heads = (medusa_head_t*)calloc(num_heads, sizeof(medusa_head_t));
    if (!model->heads) {
        free(model);
        return NULL;
    }
    
    /* Initialize each head */
    for (int i = 0; i < num_heads; i++) {
        model->heads[i].head_id = i;
        model->heads[i].hidden_size = hidden_size;
        model->heads[i].vocab_size = vocab_size;
        /* Weights allocated when loading */
    }
    
    return model;
}

/*
 * Free Medusa model
 */
void medusa_model_free(medusa_model_t* model) {
    if (!model) return;
    
    if (model->heads) {
        for (int i = 0; i < model->num_heads; i++) {
            if (model->heads[i].weights) {
                dequantized_tensor_free(model->heads[i].weights);
            }
            aligned_free(model->heads[i].bias);
        }
        free(model->heads);
    }
    
    free(model);
}

/*
 * Create candidate tree
 */
medusa_tree_t* medusa_tree_create(int max_candidates) {
    medusa_tree_t* tree = (medusa_tree_t*)calloc(1, sizeof(medusa_tree_t));
    if (!tree) return NULL;
    
    tree->tokens = (int*)malloc(max_candidates * sizeof(int));
    tree->probs = (float*)malloc(max_candidates * sizeof(float));
    tree->parent_indices = (int*)malloc(max_candidates * sizeof(int));
    tree->positions = (int*)malloc(max_candidates * sizeof(int));
    
    if (!tree->tokens || !tree->probs || !tree->parent_indices || !tree->positions) {
        medusa_tree_free(tree);
        return NULL;
    }
    
    tree->max_candidates = max_candidates;
    tree->num_candidates = 0;
    
    return tree;
}

/*
 * Free candidate tree
 */
void medusa_tree_free(medusa_tree_t* tree) {
    if (tree) {
        free(tree->tokens);
        free(tree->probs);
        free(tree->parent_indices);
        free(tree->positions);
        free(tree);
    }
}

/*
 * Simple softmax
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
 * Get top-k elements from array
 */
static void top_k(const float* values, int n, int k, int* indices, float* top_values) {
    /* Simple O(n*k) selection - sufficient for small k */
    bool* selected = (bool*)calloc(n, sizeof(bool));
    
    for (int i = 0; i < k && i < n; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int j = 0; j < n; j++) {
            if (!selected[j] && values[j] > max_val) {
                max_val = values[j];
                max_idx = j;
            }
        }
        
        if (max_idx >= 0) {
            selected[max_idx] = true;
            indices[i] = max_idx;
            top_values[i] = max_val;
        }
    }
    
    free(selected);
}

/*
 * Forward pass through Medusa head
 * Computes logits = head(hidden_state)
 */
static void medusa_head_forward(
    const medusa_head_t* head,
    const float* hidden_state,
    float* logits,
    int vocab_size,
    int hidden_size
) {
    /* Matrix multiply: [vocab] = [vocab, hidden] @ [hidden] */
    if (head->weights) {
        /* Use dequantized matmul */
        matmul_dequantized(hidden_state, head->weights, logits, 1, vocab_size, hidden_size);
        
        /* Add bias if present */
        if (head->bias) {
            for (int i = 0; i < vocab_size; i++) {
                logits[i] += head->bias[i];
            }
        }
    } else {
        /* Random logits for testing */
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    }
}

/*
 * Predict multiple tokens using Medusa heads
 * 
 * Each head predicts token at position t+head_id+1
 */
int medusa_predict(
    medusa_model_t* model,
    const float* hidden_state,
    int current_token,
    medusa_tree_t* tree,
    const medusa_config_t* config
) {
    if (!model || !tree || !config) return -1;
    
    /* Allocate logits buffer */
    float* logits = (float*)aligned_malloc(model->vocab_size * sizeof(float), 64);
    if (!logits) return -1;
    
    /* Clear tree */
    tree->num_candidates = 0;
    
    /* Add root (current token) */
    if (tree->num_candidates < tree->max_candidates) {
        tree->tokens[tree->num_candidates] = current_token;
        tree->probs[tree->num_candidates] = 1.0f;
        tree->parent_indices[tree->num_candidates] = -1;
        tree->positions[tree->num_candidates] = 0;
        tree->num_candidates++;
    }
    
    /* For each head, predict top-k tokens */
    for (int h = 0; h < model->num_heads; h++) {
        medusa_head_t* head = &model->heads[h];
        
        /* Forward through head */
        medusa_head_forward(head, hidden_state, logits, model->vocab_size, model->hidden_size);
        
        /* Apply temperature and get probabilities */
        for (int i = 0; i < model->vocab_size; i++) {
            logits[i] = logf(fmaxf(logits[i], 1e-10f)) / config->temperature;
        }
        softmax(logits, model->vocab_size);
        
        /* Get top-k tokens */
        int top_k_indices[16];
        float top_k_probs[16];
        int k = config->top_k;
        if (k > 16) k = 16;
        
        top_k(logits, model->vocab_size, k, top_k_indices, top_k_probs);
        
        /* Add to tree as children of root (simplified) */
        /* In full implementation, would build proper tree structure */
        for (int i = 0; i < k && tree->num_candidates < tree->max_candidates; i++) {
            tree->tokens[tree->num_candidates] = top_k_indices[i];
            tree->probs[tree->num_candidates] = top_k_probs[i];
            tree->parent_indices[tree->num_candidates] = 0;  /* Parent is root */
            tree->positions[tree->num_candidates] = h + 1;   /* Position t+h+1 */
            tree->num_candidates++;
        }
    }
    
    aligned_free(logits);
    
    return tree->num_candidates;
}

/*
 * Verify candidates with tree attention
 * 
 * Simplified: just accept based on probability threshold
 * Full implementation would use tree-structured attention
 */
int medusa_verify_tree(
    void* base_model,
    void (*base_forward)(void*, const int*, int, float*, int),
    const medusa_tree_t* tree,
    int* accepted_tokens,
    int* num_accepted,
    const medusa_config_t* config
) {
    if (!tree || !accepted_tokens || !num_accepted) return -1;
    
    *num_accepted = 0;
    
    /* Accept tokens with highest probabilities */
    /* In full implementation, would verify with base model */
    
    int max_accept = config->num_heads + 1;
    if (max_accept > tree->num_candidates) {
        max_accept = tree->num_candidates;
    }
    
    /* Sort by position and probability */
    /* Simplified: just accept first N distinct positions */
    int last_position = 0;
    for (int i = 1; i < tree->num_candidates && *num_accepted < max_accept; i++) {
        if (tree->positions[i] > last_position) {
            /* New position - accept with probability proportional to confidence */
            float accept_threshold = 0.5f;
            if (tree->probs[i] > accept_threshold) {
                accepted_tokens[(*num_accepted)++] = tree->tokens[i];
                last_position = tree->positions[i];
            } else {
                /* Reject - stop */
                break;
            }
        }
    }
    
    return *num_accepted;
}

/*
 * Single step of Medusa decoding
 */
int medusa_decode_step(
    void* base_model,
    void (*base_forward)(void*, const int*, int, float*, int),
    medusa_model_t* medusa,
    const float* current_hidden,
    int current_token,
    int* output_tokens,
    int* num_output,
    const medusa_config_t* config
) {
    if (!medusa || !output_tokens || !num_output || !config) return -1;
    
    /* Create tree */
    medusa_tree_t* tree = medusa_tree_create(128);
    if (!tree) return -1;
    
    /* Step 1: Predict with Medusa heads */
    int num_candidates = medusa_predict(medusa, current_hidden, current_token, tree, config);
    
    if (num_candidates <= 0) {
        medusa_tree_free(tree);
        return -1;
    }
    
    /* Step 2: Verify with tree attention */
    int num_accepted = 0;
    medusa_verify_tree(base_model, base_forward, tree, output_tokens, &num_accepted, config);
    
    *num_output = num_accepted;
    
    /* If nothing accepted, fallback to base model */
    if (num_accepted == 0) {
        *num_output = 1;
        output_tokens[0] = tree->tokens[1];  /* First prediction */
    }
    
    medusa_tree_free(tree);
    
    return *num_output;
}

/*
 * Full Medusa generation loop
 */
int medusa_generate(
    void* base_model,
    void (*base_forward)(void*, const int*, int, float*, int),
    medusa_model_t* medusa,
    const float* prompt_hidden,
    int prompt_len,
    int* output_tokens,
    int num_tokens,
    const medusa_config_t* config
) {
    if (!medusa || !output_tokens || !config) return 0;
    
    int generated = 0;
    float* current_hidden = (float*)aligned_malloc(medusa->hidden_size * sizeof(float), 64);
    
    memcpy(current_hidden, prompt_hidden, medusa->hidden_size * sizeof(float));
    
    printf("Medusa: %d heads, top_k=%d\n", config->num_heads, config->top_k);
    
    while (generated < num_tokens) {
        int step_tokens[8];
        int num_step = 0;
        
        int ret = medusa_decode_step(
            base_model, base_forward, medusa,
            current_hidden, generated > 0 ? output_tokens[generated - 1] : 0,
            step_tokens, &num_step, config
        );
        
        if (ret < 0 || num_step == 0) break;
        
        /* Copy to output */
        for (int i = 0; i < num_step && generated < num_tokens; i++) {
            output_tokens[generated++] = step_tokens[i];
        }
        
        /* Update hidden state (simplified) */
        for (int j = 0; j < medusa->hidden_size; j++) {
            current_hidden[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    
    aligned_free(current_hidden);
    
    printf("Generated %d tokens using Medusa\n", generated);
    
    return generated;
}
