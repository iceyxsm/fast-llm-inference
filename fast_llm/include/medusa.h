/*
 * Medusa Multi-Token Prediction
 * 
 * Research-based implementation:
 * - Paper: https://arxiv.org/abs/2401.10774
 * - GitHub: https://github.com/FasterDecoding/Medusa
 * 
 * Medusa adds extra decoding heads to predict multiple future tokens:
 * - Base model generates hidden state
 * - Head 0 (original): predicts token t+1
 * - Head 1: predicts token t+2
 * - Head 2: predicts token t+3
 * - etc.
 * 
 * Uses tree attention to verify candidates in parallel.
 * 
 * Expected speedup: 2-3x (can combine with speculative for 4-6x total)
 */

#ifndef MEDUSA_H
#define MEDUSA_H

#include <stdint.h>
#include <stdbool.h>
#include "dequantized_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Medusa configuration */
typedef struct {
    int num_heads;             /* Number of Medusa heads (typically 2-4) */
    int top_k;                 /* Top-k candidates per head (typically 4-10) */
    float temperature;         /* Sampling temperature */
} medusa_config_t;

/* 
 * Medusa head - predicts token at position t+head_id+1
 * Each head is a small linear layer: hidden -> vocab
 */
typedef struct {
    dequantized_tensor_t* weights;   /* [vocab_size, hidden_size] */
    float* bias;                     /* [vocab_size] optional */
    int head_id;                     /* Position offset (0 = t+1, 1 = t+2, etc.) */
    int hidden_size;                 /* Hidden dimension */
    int vocab_size;                  /* Vocab size */
} medusa_head_t;

/* 
 * Medusa model - collection of heads attached to base model
 */
typedef struct {
    medusa_head_t* heads;      /* Array of heads */
    int num_heads;
    int hidden_size;
    int vocab_size;
} medusa_model_t;

/* 
 * Candidate tree for verification
 * Organizes predictions in tree structure for efficient verification
 */
typedef struct {
    int* tokens;               /* Candidate token IDs [num_candidates] */
    float* probs;              /* Probabilities [num_candidates] */
    int* parent_indices;       /* Parent in tree [-1 for root] */
    int* positions;            /* Position in sequence */
    int num_candidates;
    int max_candidates;
} medusa_tree_t;

/* Default configuration */
static inline medusa_config_t medusa_default_config(void) {
    medusa_config_t cfg = {
        .num_heads = 3,       /* Predict t+1, t+2, t+3 */
        .top_k = 8,           /* Top-8 candidates per head */
        .temperature = 0.8f
    };
    return cfg;
}

/* Create Medusa model with N heads */
medusa_model_t* medusa_model_create(int num_heads, int hidden_size, int vocab_size);

/* Load pre-trained Medusa heads */
int medusa_model_load(medusa_model_t* model, const char* path);

/* Free Medusa model */
void medusa_model_free(medusa_model_t* model);

/* Create candidate tree */
medusa_tree_t* medusa_tree_create(int max_candidates);

/* Free candidate tree */
void medusa_tree_free(medusa_tree_t* tree);

/*
 * Predict multiple tokens using Medusa heads
 * 
 * Input: hidden state from base model [hidden_size]
 * Output: tree with candidates from all heads
 * 
 * For each head:
 *   - Compute logits: head(hidden) -> vocab
 *   - Get top-k tokens
 *   - Add to tree
 */
int medusa_predict(
    medusa_model_t* model,
    const float* hidden_state,
    int current_token,
    medusa_tree_t* tree,
    const medusa_config_t* config
);

/*
 * Build candidate tree from Medusa predictions
 * 
 * Combines predictions from all heads into tree structure
 * Root: current token
 * Level 1: head 0 predictions (t+1)
 * Level 2: head 1 predictions (t+2) as children of level 1
 * etc.
 */
int medusa_build_tree(
    medusa_model_t* model,
    const float* hidden_state,
    medusa_tree_t* tree,
    const medusa_config_t* config
);

/*
 * Verify candidates with tree attention
 * 
 * Uses tree-structured attention to verify all candidates
 * in a single forward pass (or minimal passes)
 * 
 * Returns: number of accepted tokens (0 to num_heads)
 */
int medusa_verify_tree(
    void* base_model,
    void (*base_forward)(void*, const int*, int, float*, int),
    const medusa_tree_t* tree,
    int* accepted_tokens,
    int* num_accepted,
    const medusa_config_t* config
);

/*
 * Single step of Medusa decoding
 * 
 * 1. Base model forward -> hidden state
 * 2. Medusa heads predict t+1, t+2, t+3
 * 3. Build candidate tree
 * 4. Verify with tree attention
 * 5. Accept longest valid prefix
 * 
 * Returns: number of tokens generated (1 to num_heads+1)
 */
int medusa_decode_step(
    /* Base model */
    void* base_model,
    void (*base_forward)(void*, const int*, int, float*, int),
    
    /* Medusa heads */
    medusa_model_t* medusa,
    
    /* Current state */
    const float* current_hidden,
    int current_token,
    
    /* Output */
    int* output_tokens,
    int* num_output,
    
    /* Config */
    const medusa_config_t* config
);

/*
 * Full Medusa generation loop
 * 
 * Generates num_tokens using Medusa multi-token prediction
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
);

/*
 * Combined Speculative + Medusa
 * 
 * Ultimate speedup combination:
 * 1. Draft model generates candidates (EAGLE-style)
 * 2. Medusa heads verify and extend (Medusa-style)
 * 3. Target model verifies final candidates
 * 
 * Expected: 4-6x speedup
 */
int speculative_medusa_generate(
    /* Draft model (EAGLE) */
    void* draft_model,
    void (*draft_forward)(void*, const float*, float*, int),
    
    /* Medusa heads */
    medusa_model_t* medusa,
    
    /* Target model */
    void* target_model,
    void (*target_forward)(void*, const int*, int, float*, int),
    
    const float* prompt_hidden,
    int prompt_len,
    
    int* output_tokens,
    int num_tokens,
    
    const medusa_config_t* config
);

#ifdef __cplusplus
}
#endif

#endif /* MEDUSA_H */
