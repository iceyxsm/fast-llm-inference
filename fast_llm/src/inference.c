/*
 * Inference Engine - Single forward pass and generation
 * OPTIMIZED VERSION: Uses 6x16 ASM-style kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

/* Use optimized kernels */
#define USE_OPTIMIZED_MATMUL 1

/* Global: max layers to use (0 = all) */
int g_max_layers = 0;

/* Enable prefetching for better memory performance */
#define USE_PREFETCHING 1

/* Enable non-temporal stores for large outputs */
#define USE_STREAMING_STORES 1

#define GELU_SCALING 0.7978845608f  /* sqrt(2/pi) */

/* GELU activation with tanh approximation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(GELU_SCALING * (x + 0.044715f * x * x * x)));
}

/* AVX2-optimized activations */
extern void silu_avx2(const float* input, float* output, int n);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

/* Silu activation: x * sigmoid(x) */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* RMS norm: x / sqrt(mean(x^2) + eps) */
static void rms_norm(const float* x, float* out, int n, float eps) {
    #ifdef __AVX2__
    rms_norm_avx2(x, out, n, eps);
    #else
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
    #endif
}

/* Softmax: exp(x_i - max) / sum(exp(x_j - max)) */
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/* Simple random sampling with temperature */
static int sample_token(float* logits, int vocab_size, float temperature) {
    /* Apply temperature */
    if (temperature != 1.0f && temperature > 0) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }
    
    /* Convert to probabilities */
    softmax(logits, vocab_size);
    
    /* Simple argmax for now (greedy) */
    int max_idx = 0;
    float max_prob = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_prob) {
            max_prob = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/* Matmul with regular float weights (fallback) */
static void matmul_float(const float* A, const float* B, float* C, 
                         int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];  /* B is column-major */
            }
            C[i * N + j] = sum;
        }
    }
}

/* Attention computation: Q @ K^T / sqrt(d_k) */
static void compute_attention(const float* Q, const float* K, const float* V,
                              float* output, int seq_len, int num_heads, int head_dim) {
    int head_size = head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    #pragma omp parallel for
    for (int h = 0; h < num_heads; h++) {
        const float* Q_h = Q + h * head_size;
        const float* K_h = K + h * head_size;
        const float* V_h = V + h * head_size;
        float* out_h = output + h * head_size;
        
        /* Compute Q @ K^T for this head */
        float scores[4096]; /* Max seq len */
        
        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                float dot = 0.0f;
                for (int d = 0; d < head_size; d++) {
                    dot += Q_h[q_pos * num_heads * head_size + d] * 
                           K_h[k_pos * num_heads * head_size + d];
                }
                scores[k_pos] = dot * scale;
            }
            
            /* Apply causal mask (already handled by k_pos <= q_pos) */
            /* Softmax over attended positions */
            float max_score = scores[0];
            for (int k = 1; k <= q_pos; k++) {
                if (scores[k] > max_score) max_score = scores[k];
            }
            
            float sum = 0.0f;
            for (int k = 0; k <= q_pos; k++) {
                scores[k] = expf(scores[k] - max_score);
                sum += scores[k];
            }
            
            for (int k = 0; k <= q_pos; k++) {
                scores[k] /= sum;
            }
            
            /* Weighted sum of values */
            for (int d = 0; d < head_size; d++) {
                float val = 0.0f;
                for (int k = 0; k <= q_pos; k++) {
                    val += scores[k] * V_h[k * num_heads * head_size + d];
                }
                out_h[q_pos * num_heads * head_size + d] = val;
            }
        }
    }
}

/* Single transformer layer forward pass */
static void transformer_layer(transformer_model_t* model, int layer_idx,
                               float* hidden_states, int seq_len, int batch_size) {
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    int num_heads = model->config.num_heads;
    int head_dim = model->config.head_dim;
    
    /* Allocate temporaries */
    float* residual = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* q = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* k = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* v = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* attn_out = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* ff_normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* gate = aligned_malloc(seq_len * intermediate_size * sizeof(float), 32);
    float* up = aligned_malloc(seq_len * intermediate_size * sizeof(float), 32);
    float* ff_out = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    
    memcpy(residual, hidden_states, seq_len * hidden_size * sizeof(float));
    
    /* Pre-attention RMS norm */
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &normed[s * hidden_size], 
                 hidden_size, 1e-5f);
    }
    
    /* Q, K, V projections using optimized INT8 matmul kernels */
    for (int s = 0; s < seq_len; s++) {
        float* norm_row = &normed[s * hidden_size];
        float* q_row = &q[s * hidden_size];
        float* k_row = &k[s * hidden_size];
        float* v_row = &v[s * hidden_size];
        
        /* Q projection: [hidden] @ [hidden, hidden] -> [hidden] */
        if (model->q_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            matmul_dequantized_asm_style(norm_row, model->q_proj[layer_idx], q_row, 1, hidden_size, hidden_size);
            #else
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                const int8_t* w = model->q_proj[layer_idx]->weights;
                float scale = model->q_proj[layer_idx]->scales[h];
                for (int d = 0; d < hidden_size; d++) {
                    sum += norm_row[d] * w[h * hidden_size + d] * scale;
                }
                q_row[h] = sum;
            }
            #endif
        }
        
        /* K projection */
        if (model->k_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            matmul_dequantized_asm_style(norm_row, model->k_proj[layer_idx], k_row, 1, hidden_size, hidden_size);
            #else
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                const int8_t* w = model->k_proj[layer_idx]->weights;
                float scale = model->k_proj[layer_idx]->scales[h];
                for (int d = 0; d < hidden_size; d++) {
                    sum += norm_row[d] * w[h * hidden_size + d] * scale;
                }
                k_row[h] = sum;
            }
            #endif
        }
        
        /* V projection */
        if (model->v_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            matmul_dequantized_asm_style(norm_row, model->v_proj[layer_idx], v_row, 1, hidden_size, hidden_size);
            #else
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                const int8_t* w = model->v_proj[layer_idx]->weights;
                float scale = model->v_proj[layer_idx]->scales[h];
                for (int d = 0; d < hidden_size; d++) {
                    sum += norm_row[d] * w[h * hidden_size + d] * scale;
                }
                v_row[h] = sum;
            }
            #endif
        }
    }
    
    /* Multi-head attention */
    compute_attention(q, k, v, attn_out, seq_len, num_heads, head_dim);
    
    /* O projection using optimized kernel */
    for (int s = 0; s < seq_len; s++) {
        float* attn_row = &attn_out[s * hidden_size];
        float* out_row = &hidden_states[s * hidden_size];
        float* res_row = &residual[s * hidden_size];
        
        if (model->o_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            /* Compute O projection: [hidden] @ [hidden, hidden] -> [hidden] */
            float o_out[4096]; /* Max hidden size */
            matmul_dequantized_asm_style(attn_row, model->o_proj[layer_idx], o_out, 1, hidden_size, hidden_size);
            /* Add residual */
            for (int h = 0; h < hidden_size; h++) {
                out_row[h] = res_row[h] + o_out[h];
            }
            #else
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                const int8_t* w = model->o_proj[layer_idx]->weights;
                float scale = model->o_proj[layer_idx]->scales[h];
                for (int d = 0; d < hidden_size; d++) {
                    sum += attn_row[d] * w[h * hidden_size + d] * scale;
                }
                out_row[h] = res_row[h] + sum;
            }
            #endif
        } else {
            for (int h = 0; h < hidden_size; h++) {
                out_row[h] = res_row[h];
            }
        }
    }
    
    /* Save residual for FFN */
    memcpy(residual, hidden_states, seq_len * hidden_size * sizeof(float));
    
    /* Post-attention RMS norm */
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &ff_normed[s * hidden_size],
                 hidden_size, 1e-5f);
    }
    
    /* FFN: gate_proj and up_proj using optimized kernels */
    for (int s = 0; s < seq_len; s++) {
        float* ff_row = &ff_normed[s * hidden_size];
        float* gate_row = &gate[s * intermediate_size];
        float* up_row = &up[s * intermediate_size];
        
        /* Gate projection (SiLU) */
        if (model->gate_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            matmul_dequantized_asm_style(ff_row, model->gate_proj[layer_idx], gate_row, 1, intermediate_size, hidden_size);
            /* SiLU will be applied in fused SwiGLU below */
            #else
            for (int i = 0; i < intermediate_size; i++) {
                float sum = 0.0f;
                const int8_t* w = model->gate_proj[layer_idx]->weights;
                float scale = model->gate_proj[layer_idx]->scales[i];
                for (int d = 0; d < hidden_size; d++) {
                    sum += ff_row[d] * w[i * hidden_size + d] * scale;
                }
                gate_row[i] = silu(sum);
            }
            #endif
        }
        
        /* Up projection */
        if (model->up_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            matmul_dequantized_asm_style(ff_row, model->up_proj[layer_idx], up_row, 1, intermediate_size, hidden_size);
            #else
            for (int i = 0; i < intermediate_size; i++) {
                float sum = 0.0f;
                const int8_t* w = model->up_proj[layer_idx]->weights;
                float scale = model->up_proj[layer_idx]->scales[i];
                for (int d = 0; d < hidden_size; d++) {
                    sum += ff_row[d] * w[i * hidden_size + d] * scale;
                }
                up_row[i] = sum;
            }
            #endif
        }
        
        /* SwiGLU: fused SiLU(gate) * up using AVX2 */
        #ifdef __AVX2__
        swiglu_avx2(gate_row, up_row, gate_row, intermediate_size);
        #else
        for (int i = 0; i < intermediate_size; i++) {
            gate_row[i] = silu(gate_row[i]) * up_row[i];
        }
        #endif
    }
    
    /* Down projection using optimized kernel */
    for (int s = 0; s < seq_len; s++) {
        float* gate_row = &gate[s * intermediate_size];
        float* out_row = &hidden_states[s * hidden_size];
        float* res_row = &residual[s * hidden_size];
        
        if (model->down_proj[layer_idx]) {
            #if USE_OPTIMIZED_MATMUL
            /* Compute down projection: [intermediate] @ [hidden, intermediate] -> [hidden] */
            float down_out[4096]; /* Max hidden size */
            matmul_dequantized_asm_style(gate_row, model->down_proj[layer_idx], down_out, 1, hidden_size, intermediate_size);
            /* Add residual */
            for (int h = 0; h < hidden_size; h++) {
                out_row[h] = res_row[h] + down_out[h];
            }
            #else
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                const int8_t* w = model->down_proj[layer_idx]->weights;
                float scale = model->down_proj[layer_idx]->scales[h];
                for (int i = 0; i < intermediate_size; i++) {
                    sum += gate_row[i] * w[h * intermediate_size + i] * scale;
                }
                out_row[h] = res_row[h] + sum;
            }
            #endif
        } else {
            for (int h = 0; h < hidden_size; h++) {
                out_row[h] = res_row[h];
            }
        }
    }
    
    /* Cleanup */
    aligned_free(residual);
    aligned_free(normed);
    aligned_free(q);
    aligned_free(k);
    aligned_free(v);
    aligned_free(attn_out);
    aligned_free(ff_normed);
    aligned_free(gate);
    aligned_free(up);
    aligned_free(ff_out);
}

/* Full forward pass through all layers */
void model_forward(transformer_model_t* model,
                   const int* input_tokens, int seq_len,
                   float* output_logits, int* output_tokens) {
    int hidden_size = model->config.hidden_size;
    int vocab_size = model->config.vocab_size;
    int num_layers = model->config.num_layers;
    
    /* Apply layer reduction if configured */
    if (g_max_layers > 0 && g_max_layers < num_layers) {
        num_layers = g_max_layers;
    }
    
    /* Allocate hidden states */
    float* hidden_states = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    
    /* Embedding lookup */
    for (int s = 0; s < seq_len; s++) {
        int token = input_tokens[s];
        if (token >= vocab_size) token = 0; /* Bounds check */
        memcpy(&hidden_states[s * hidden_size],
               &model->token_embeddings[token * hidden_size],
               hidden_size * sizeof(float));
    }
    
    /* Run through all transformer layers */
    for (int layer = 0; layer < num_layers; layer++) {
        transformer_layer(model, layer, hidden_states, seq_len, 1);
    }
    
    /* Final RMS norm (Phi-3 style) */
    float* normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &normed[s * hidden_size],
                 hidden_size, 1e-5f);
    }
    
    /* LM head projection using optimized kernel */
    float* logits = aligned_malloc(vocab_size * sizeof(float), 32);
    
    /* Use last token for next token prediction */
    float* last_hidden = &normed[(seq_len - 1) * hidden_size];
    
    if (model->lm_head) {
        #if USE_OPTIMIZED_MATMUL
        matmul_dequantized_asm_style(last_hidden, model->lm_head, logits, 1, vocab_size, hidden_size);
        #else
        for (int v = 0; v < vocab_size; v++) {
            float sum = 0.0f;
            const int8_t* w = model->lm_head->weights;
            float scale = model->lm_head->scales[v];
            for (int d = 0; d < hidden_size; d++) {
                sum += last_hidden[d] * w[v * hidden_size + d] * scale;
            }
            logits[v] = sum;
        }
        #endif
    } else {
        memset(logits, 0, vocab_size * sizeof(float));
    }
    
    /* Copy to output */
    if (output_logits) {
        memcpy(output_logits, logits, vocab_size * sizeof(float));
    }
    
    /* Sample next token */
    if (output_tokens) {
        output_tokens[0] = sample_token(logits, vocab_size, 0.8f);
    }
    
    aligned_free(hidden_states);
    aligned_free(normed);
    aligned_free(logits);
}

/* Generate tokens autoregressively */
int model_generate(transformer_model_t* model,
                   const char* prompt,
                   int* output_tokens,
                   int max_tokens,
                   float temperature,
                   bool use_speculative,
                   bool use_medusa) {
    (void)prompt; /* TODO: Implement tokenizer */
    (void)use_speculative;
    (void)use_medusa;
    
    /* For now, use random starting tokens */
    int seq_len = 1;
    int* input_tokens = calloc(128, sizeof(int));
    input_tokens[0] = 1; /* BOS token placeholder */
    
    float* logits = aligned_malloc(model->config.vocab_size * sizeof(float), 32);
    int next_token;
    
    printf("Generating %d tokens...\n", max_tokens);
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    
    for (int i = 0; i < max_tokens; i++) {
        model_forward(model, input_tokens, seq_len, logits, &next_token);
        
        output_tokens[i] = next_token;
        input_tokens[seq_len] = next_token;
        seq_len++;
        
        if (seq_len >= 128) {
            /* Shift context window */
            memmove(input_tokens, input_tokens + 64, (seq_len - 64) * sizeof(int));
            seq_len -= 64;
        }
        
        /* Stop on EOS token (placeholder) */
        if (next_token == 2) {
            printf("<EOS>\n");
            break;
        }
    }
    
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    double elapsed = end_time - start_time;
    double tokens_per_sec = max_tokens / elapsed;
    
    printf("\nGenerated %d tokens in %.2f seconds\n", max_tokens, elapsed);
    printf("Speed: %.2f tokens/second\n", tokens_per_sec);
    
    free(input_tokens);
    aligned_free(logits);
    
    return max_tokens;
}

/* Benchmark model speed */
double model_benchmark(transformer_model_t* model, int num_tokens, bool use_optimizations) {
    (void)use_optimizations; /* Optimizations always on via INT8 */
    
    int hidden_size = model->config.hidden_size;
    int vocab_size = model->config.vocab_size;
    
    /* Create random input */
    int* tokens = calloc(num_tokens, sizeof(int));
    for (int i = 0; i < num_tokens; i++) {
        tokens[i] = rand() % vocab_size;
    }
    
    float* logits = aligned_malloc(vocab_size * sizeof(float), 32);
    int next_token;
    
    /* Warmup */
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        model_forward(model, tokens, 1, logits, &next_token);
    }
    
    /* Benchmark */
    printf("Running benchmark with %d tokens...\n", num_tokens);
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    
    for (int i = 0; i < num_tokens; i++) {
        int seq_len = (i < 10) ? i + 1 : 10; /* Keep seq len reasonable */
        model_forward(model, tokens, seq_len, logits, &next_token);
        tokens[i % 10] = next_token; /* Update input */
    }
    
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    double elapsed = end_time - start_time;
    double tokens_per_sec = num_tokens / elapsed;
    
    printf("\n=== Benchmark Results ===\n");
    printf("Tokens: %d\n", num_tokens);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tokens_per_sec);
    printf("Time per token: %.2f ms\n", 1000.0 / tokens_per_sec);
    printf("=========================\n\n");
    
    free(tokens);
    aligned_free(logits);
    
    return tokens_per_sec;
}
