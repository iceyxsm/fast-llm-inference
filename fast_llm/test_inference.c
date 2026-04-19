/* Quick test: load model, tokenize, run forward, print results */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model_loader.h"
#include "dequantized_tensor.h"
#include "cpu_features.h"

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define aligned_malloc(sz, al) _aligned_malloc(sz, al)
#define aligned_free(p) _aligned_free(p)
#else
#define aligned_malloc(sz, al) aligned_alloc(al, sz)
#define aligned_free(p) free(p)
#endif

extern void model_forward(transformer_model_t* model,
                          const int* input_tokens, int seq_len,
                          float* output_logits, int* output_tokens);
extern int g_max_layers;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: test_inference <model.gguf>\n");
        return 1;
    }

#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    printf("=== INFERENCE TEST ===\n\n");

    /* Load model */
    printf("Loading: %s\n", argv[1]);
    transformer_model_t* model = model_load_gguf(argv[1], 1);
    if (!model) { printf("FAILED to load model!\n"); return 1; }

    g_max_layers = model->config.num_layers;

    printf("\nModel loaded:\n");
    printf("  vocab_size: %d\n", model->config.vocab_size);
    printf("  hidden_size: %d\n", model->config.hidden_size);
    printf("  num_layers: %d\n", model->config.num_layers);
    printf("  num_heads: %d / %d\n", model->config.num_heads, model->config.num_kv_heads);
    printf("  vocab_loaded: %d\n", model->vocab_loaded);

    /* Check if key weights are non-null */
    printf("\nWeight check:\n");
    printf("  token_embeddings: %s\n", model->token_embeddings ? "OK" : "NULL");
    printf("  lm_head: %s\n", model->lm_head ? "OK" : "NULL");
    if (model->lm_head) printf("    lm_head dims: %d x %d\n", model->lm_head->rows, model->lm_head->cols);

    int ok_layers = 0;
    for (int l = 0; l < model->config.num_layers; l++) {
        if (model->gate_proj[l] && model->up_proj[l] && model->down_proj[l] &&
            model->q_proj[l] && model->k_proj[l] && model->v_proj[l] && model->o_proj[l])
            ok_layers++;
    }
    printf("  Complete layers: %d / %d\n", ok_layers, model->config.num_layers);

    int norm_ok = 0;
    for (int l = 0; l < model->config.num_layers; l++)
        if (model->input_layernorm[l] && model->post_attn_layernorm[l]) norm_ok++;
    printf("  Layers with norms: %d / %d\n", norm_ok, model->config.num_layers);

    /* Check embedding values */
    printf("\nEmbedding sample (token 1, first 5 values):\n  ");
    for (int i = 0; i < 5 && i < model->config.hidden_size; i++)
        printf("%.6f ", model->token_embeddings[1 * model->config.hidden_size + i]);
    printf("\n");

    /* Simple forward pass with token IDs [1, 15339] = BOS + "hello" in llama */
    int input[] = {1, 15339};
    int seq_len = 2;
    int vocab = model->config.vocab_size;

    printf("Running forward pass (input: [1, 15339], seq_len=%d, vocab=%d)...\n", seq_len, vocab);
    printf("  Embedding lookup...\n"); fflush(stdout);

    float* logits = (float*)aligned_malloc(vocab * sizeof(float), 64);
    int next_token = 0;

    clock_t t0 = clock();
    model_forward(model, input, seq_len, logits, &next_token);
    clock_t t1 = clock();
    double ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    printf("Forward pass done in %.0f ms\n", ms);
    printf("Sampled token: %d\n", next_token);

    /* Show top 10 logits */
    printf("\nTop 10 logits:\n");
    int top_ids[10] = {0};
    float top_vals[10] = {0};
    for (int v = 0; v < vocab; v++) {
        for (int t = 0; t < 10; t++) {
            if (logits[v] > top_vals[t]) {
                for (int s = 9; s > t; s--) { top_ids[s] = top_ids[s-1]; top_vals[s] = top_vals[s-1]; }
                top_ids[t] = v;
                top_vals[t] = logits[v];
                break;
            }
        }
    }
    for (int t = 0; t < 10; t++) {
        printf("  [%d] token=%d logit=%.4f", t+1, top_ids[t], top_vals[t]);
        if (model->vocab_loaded && top_ids[t] < vocab && model->vocab_tokens[top_ids[t]])
            printf(" '%s'", model->vocab_tokens[top_ids[t]]);
        printf("\n");
    }

    /* Check if logits are all zeros */
    float sum = 0;
    for (int v = 0; v < vocab; v++) sum += logits[v] * logits[v];
    printf("\nLogits L2 norm: %.6f %s\n", sum, sum < 0.001 ? "(ALL ZEROS - weights not loaded!)" : "(non-zero - OK)");

    aligned_free(logits);
    model_free(model);
    printf("\nDone.\n");
    return 0;
}
