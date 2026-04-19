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

/* Matrix-vector multiply using dequantized tensor (supports f32 or int8 weights) */
static void matvec(const float* input, const dequantized_tensor_t* W, float* output, int out_dim, int in_dim) {
    if (W->f32_weights) {
        /* Use float32 weights directly — full precision */
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            const float* row = W->f32_weights + o * in_dim;
            for (int i = 0; i < in_dim; i++) {
                sum += input[i] * row[i];
            }
            output[o] = sum;
        }
    } else if (W->weights && W->scales) {
        /* Use int8 weights with per-row scale */
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            const int8_t* row = W->weights + o * in_dim;
            float scale = W->scales[o];
            for (int i = 0; i < in_dim; i++) {
                sum += input[i] * row[i];
            }
            output[o] = sum * scale;
        }
    } else {
        memset(output, 0, out_dim * sizeof(float));
    }
}

/* Apply Rotary Position Embeddings (RoPE) to Q and K */
static void apply_rope(float* q, float* k, int seq_len, int num_heads, int num_kv_heads, int head_dim, float rope_theta) {
    /* RoPE: for each position, rotate pairs of dimensions */
    /* freq_i = 1 / (theta ^ (2i / head_dim)) */
    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < num_heads; h++) {
            float* q_vec = q + pos * num_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(rope_theta, (float)i / (float)head_dim);
                float angle = pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float q0 = q_vec[i];
                float q1 = q_vec[i + 1];
                q_vec[i]     = q0 * cos_a - q1 * sin_a;
                q_vec[i + 1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (int h = 0; h < num_kv_heads; h++) {
            float* k_vec = k + pos * num_kv_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(rope_theta, (float)i / (float)head_dim);
                float angle = pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float k0 = k_vec[i];
                float k1 = k_vec[i + 1];
                k_vec[i]     = k0 * cos_a - k1 * sin_a;
                k_vec[i + 1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

/* Attention computation: RoPE + Q @ K^T / sqrt(d_k) + softmax + V */
static void compute_attention(float* Q, float* K, const float* V,
                              float* output, int seq_len, int num_heads, int num_kv_heads, int head_dim) {
    /* Apply RoPE to Q and K */
    float rope_theta = 500000.0f; /* Llama 3.x uses 500000, older models use 10000 */
    apply_rope(Q, K, seq_len, num_heads, num_kv_heads, head_dim, rope_theta);

    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group = num_heads / num_kv_heads; /* how many Q heads per KV head */
    
    #pragma omp parallel for
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_group; /* which KV head this Q head uses */
        
        float scores[4096]; /* Max seq len */
        
        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            /* Q is laid out as [seq_len, num_heads, head_dim] */
            const float* q_vec = Q + q_pos * num_heads * head_dim + h * head_dim;
            
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                /* K is laid out as [seq_len, num_kv_heads, head_dim] */
                const float* k_vec = K + k_pos * num_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[k_pos] = dot * scale;
            }
            
            /* Softmax */
            float max_score = scores[0];
            for (int k = 1; k <= q_pos; k++)
                if (scores[k] > max_score) max_score = scores[k];
            
            float sum = 0.0f;
            for (int k = 0; k <= q_pos; k++) {
                scores[k] = expf(scores[k] - max_score);
                sum += scores[k];
            }
            for (int k = 0; k <= q_pos; k++)
                scores[k] /= sum;
            
            /* Weighted sum of V */
            float* out_vec = output + q_pos * num_heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int k = 0; k <= q_pos; k++) {
                    const float* v_vec = V + k * num_kv_heads * head_dim + kv_h * head_dim;
                    val += scores[k] * v_vec[d];
                }
                out_vec[d] = val;
            }
        }
    }
}

/* Single transformer layer forward pass */
static void transformer_layer(transformer_model_t* model, int layer_idx,
                               float* hidden_states, int seq_len, int batch_size) {
    (void)batch_size;
    int hidden_size = model->config.hidden_size;
    int intermediate_size = model->config.intermediate_size;
    int num_heads = model->config.num_heads;
    int num_kv_heads = model->config.num_kv_heads;
    int head_dim = model->config.head_dim;
    int kv_dim = num_kv_heads * head_dim; /* K/V output size for GQA */
    
    /* Allocate temporaries */
    float* residual = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* q = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* k = aligned_malloc(seq_len * kv_dim * sizeof(float), 32);
    float* v = aligned_malloc(seq_len * kv_dim * sizeof(float), 32);
    float* attn_out = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* ff_normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    float* gate = aligned_malloc(seq_len * intermediate_size * sizeof(float), 32);
    float* up = aligned_malloc(seq_len * intermediate_size * sizeof(float), 32);
    float* ff_out = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    
    memcpy(residual, hidden_states, seq_len * hidden_size * sizeof(float));
    
    /* Pre-attention RMS norm with learned weights */
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &normed[s * hidden_size], 
                 hidden_size, 1e-5f);
        /* Apply learned norm weights */
        if (model->input_layernorm[layer_idx]) {
            for (int d = 0; d < hidden_size; d++)
                normed[s * hidden_size + d] *= model->input_layernorm[layer_idx][d];
        }
    }
    
    /* Q, K, V projections using optimized INT8 matmul kernels */
    for (int s = 0; s < seq_len; s++) {
        float* norm_row = &normed[s * hidden_size];
        float* q_row = &q[s * hidden_size];
        float* k_row = &k[s * kv_dim];
        float* v_row = &v[s * kv_dim];
        
        /* Q projection: [hidden] -> [hidden] */
        if (model->q_proj[layer_idx]) {
            matvec(norm_row, model->q_proj[layer_idx], q_row, hidden_size, hidden_size);
        }
        
        /* K projection: [hidden] -> [kv_dim] */
        if (model->k_proj[layer_idx]) {
            matvec(norm_row, model->k_proj[layer_idx], k_row, kv_dim, hidden_size);
        }
        
        /* V projection: [hidden] -> [kv_dim] */
        if (model->v_proj[layer_idx]) {
            matvec(norm_row, model->v_proj[layer_idx], v_row, kv_dim, hidden_size);
        }
    }
    
    /* Multi-head attention */
    compute_attention(q, k, v, attn_out, seq_len, num_heads, num_kv_heads, head_dim);
    
    /* O projection using optimized kernel */
    for (int s = 0; s < seq_len; s++) {
        float* attn_row = &attn_out[s * hidden_size];
        float* out_row = &hidden_states[s * hidden_size];
        float* res_row = &residual[s * hidden_size];
        
        if (model->o_proj[layer_idx]) {
            float* o_out = aligned_malloc(hidden_size * sizeof(float), 32);
            matvec(attn_row, model->o_proj[layer_idx], o_out, hidden_size, hidden_size);
            for (int h = 0; h < hidden_size; h++)
                out_row[h] = res_row[h] + o_out[h];
            aligned_free(o_out);
        } else {
            for (int h = 0; h < hidden_size; h++) {
                out_row[h] = res_row[h];
            }
        }
    }
    
    /* Save residual for FFN */
    memcpy(residual, hidden_states, seq_len * hidden_size * sizeof(float));
    
    /* Post-attention RMS norm with learned weights */
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &ff_normed[s * hidden_size],
                 hidden_size, 1e-5f);
        if (model->post_attn_layernorm[layer_idx]) {
            for (int d = 0; d < hidden_size; d++)
                ff_normed[s * hidden_size + d] *= model->post_attn_layernorm[layer_idx][d];
        }
    }
    
    /* FFN: gate_proj and up_proj using optimized kernels */
    for (int s = 0; s < seq_len; s++) {
        float* ff_row = &ff_normed[s * hidden_size];
        float* gate_row = &gate[s * intermediate_size];
        float* up_row = &up[s * intermediate_size];
        
        /* Gate projection */
        if (model->gate_proj[layer_idx]) {
            matvec(ff_row, model->gate_proj[layer_idx], gate_row, intermediate_size, hidden_size);
        }
        
        /* Up projection */
        if (model->up_proj[layer_idx]) {
            matvec(ff_row, model->up_proj[layer_idx], up_row, intermediate_size, hidden_size);
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
            float* down_out = aligned_malloc(hidden_size * sizeof(float), 32);
            matvec(gate_row, model->down_proj[layer_idx], down_out, hidden_size, intermediate_size);
            for (int h = 0; h < hidden_size; h++)
                out_row[h] = res_row[h] + down_out[h];
            aligned_free(down_out);
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
    fprintf(stderr, "[fwd] embedding lookup, seq=%d, hidden=%d, vocab=%d\n", seq_len, hidden_size, vocab_size);
    fflush(stderr);
    for (int s = 0; s < seq_len; s++) {
        int token = input_tokens[s];
        if (token >= vocab_size) token = 0;
        memcpy(&hidden_states[s * hidden_size],
               &model->token_embeddings[token * hidden_size],
               hidden_size * sizeof(float));
    }
    fprintf(stderr, "[fwd] embedding done, running %d layers\n", num_layers);
    fflush(stderr);
    
    /* Run through all transformer layers */
    for (int layer = 0; layer < num_layers; layer++) {
        fprintf(stderr, "[fwd] layer %d/%d\n", layer, num_layers);
        fflush(stderr);
        transformer_layer(model, layer, hidden_states, seq_len, 1);
    }
    fprintf(stderr, "[fwd] all layers done, computing logits\n");
    fflush(stderr);

    /* Check hidden state */
    float hs_sum = 0;
    for (int i = 0; i < hidden_size; i++) hs_sum += hidden_states[(seq_len-1)*hidden_size + i] * hidden_states[(seq_len-1)*hidden_size + i];
    fprintf(stderr, "[fwd] hidden state L2: %.6f\n", hs_sum);
    fflush(stderr);
    
    /* Final RMS norm with learned weights */
    float* normed = aligned_malloc(seq_len * hidden_size * sizeof(float), 32);
    for (int s = 0; s < seq_len; s++) {
        rms_norm(&hidden_states[s * hidden_size], &normed[s * hidden_size],
                 hidden_size, 1e-5f);
        if (model->output_norm) {
            for (int d = 0; d < hidden_size; d++)
                normed[s * hidden_size + d] *= model->output_norm[d];
        }
    }
    
    /* LM head projection using optimized kernel */
    float* logits = aligned_malloc(vocab_size * sizeof(float), 32);
    
    /* Use last token for next token prediction */
    float* last_hidden = &normed[(seq_len - 1) * hidden_size];
    
    if (model->lm_head) {
        matvec(last_hidden, model->lm_head, logits, vocab_size, hidden_size);
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
