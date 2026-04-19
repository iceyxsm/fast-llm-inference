/*
 * Interactive Chat with REAL Inference
 * 
 * Uses the model's own vocabulary from GGUF for tokenization/detokenization.
 * Applies proper chat templates based on the model architecture.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "model_loader.h"
#include "dequantized_tensor.h"

/* External real inference functions from inference.c */
extern void model_forward(transformer_model_t* model,
                          const int* input_tokens, int seq_len,
                          float* output_logits, int* output_tokens);
extern int g_max_layers;

/* ── SentencePiece-aware tokenizer ────────────────────────────────── */

#define SP_SPACE "\xE2\x96\x81"
#define SP_SPACE_LEN 3

/* Convert raw text to SentencePiece form: spaces become ▁ */
static int sp_normalize(const char* text, char* out, int osz) {
    int pos = 0;
    const char* p = text;
    if (*p && *p != ' ') {
        if (pos + SP_SPACE_LEN < osz) { memcpy(out + pos, SP_SPACE, SP_SPACE_LEN); pos += SP_SPACE_LEN; }
    }
    while (*p && pos < osz - 4) {
        if (*p == ' ') { memcpy(out + pos, SP_SPACE, SP_SPACE_LEN); pos += SP_SPACE_LEN; p++; }
        else { out[pos++] = *p++; }
    }
    out[pos] = '\0';
    return pos;
}

static int find_special_token(transformer_model_t* model, const char* text) {
    if (!model->vocab_loaded || !model->vocab_tokens) return -1;
    for (int v = 0; v < model->config.vocab_size; v++) {
        const char* tok = model->vocab_tokens[v];
        if (tok && strcmp(tok, text) == 0) return v;
    }
    return -1;
}

static int append_special(transformer_model_t* model, const char* tag, int* tokens, int count, int max) {
    if (count >= max) return count;
    int id = find_special_token(model, tag);
    if (id >= 0) tokens[count++] = id;
    return count;
}

/* Greedy longest-match tokenization on normalized text */
static int tokenize_raw(transformer_model_t* model, const char* text, int* tokens, int max_tokens) {
    int count = 0;
    if (!model->vocab_loaded || !model->vocab_tokens) {
        const unsigned char* p = (const unsigned char*)text;
        while (*p && count < max_tokens) { tokens[count++] = 3 + *p; p++; }
        return count;
    }
    int norm_sz = (int)strlen(text) * 4 + 16;
    char* norm = malloc(norm_sz);
    sp_normalize(text, norm, norm_sz);
    int vocab_size = model->config.vocab_size;
    const char* p = norm;
    while (*p && count < max_tokens) {
        int best_len = 0, best_id = -1;
        for (int v = 0; v < vocab_size; v++) {
            const char* tok = model->vocab_tokens[v];
            if (!tok || !tok[0]) continue;
            if ((unsigned char)tok[0] != (unsigned char)*p) continue;
            int tlen = (int)strlen(tok);
            if (tlen > best_len && strncmp(p, tok, tlen) == 0) { best_len = tlen; best_id = v; }
        }
        if (best_len > 0 && best_id >= 0) { tokens[count++] = best_id; p += best_len; }
        else {
            unsigned char ch = (unsigned char)*p;
            int found = 0;
            for (int v = 0; v < vocab_size; v++) {
                const char* tok = model->vocab_tokens[v];
                if (tok && strlen(tok) == 1 && (unsigned char)tok[0] == ch) { tokens[count++] = v; found = 1; break; }
            }
            if (!found) tokens[count++] = 3 + ch;
            p++;
        }
    }
    free(norm);
    return count;
}

/* Detect chat template: 1=Phi3/Llama3, 2=ChatML, 3=Llama2, 0=none */
static int detect_template(transformer_model_t* model) {
    if (!model->vocab_loaded || !model->vocab_tokens) return 0;
    if (find_special_token(model, "<|user|>") >= 0 || find_special_token(model, "<|start_header_id|>") >= 0) return 1;
    if (find_special_token(model, "<|im_start|>") >= 0) return 2;
    if (find_special_token(model, "[INST]") >= 0) return 3;
    return 0;
}

/* Full tokenization with chat template */
static int tokenize_chat(transformer_model_t* model, const char* user_msg, int* tokens, int max_tokens) {
    int count = 0;
    tokens[count++] = 1; /* BOS */
    int tmpl = detect_template(model);

    if (tmpl == 1) {
        if (find_special_token(model, "<|user|>") >= 0) {
            count = append_special(model, "<|user|>", tokens, count, max_tokens);
            char* w = malloc(strlen(user_msg) + 4); sprintf(w, "\n%s", user_msg);
            count += tokenize_raw(model, w, tokens + count, max_tokens - count); free(w);
            count = append_special(model, "<|end|>", tokens, count, max_tokens);
            count = append_special(model, "<|assistant|>", tokens, count, max_tokens);
        } else {
            count = append_special(model, "<|start_header_id|>", tokens, count, max_tokens);
            count += tokenize_raw(model, "user", tokens + count, max_tokens - count);
            count = append_special(model, "<|end_header_id|>", tokens, count, max_tokens);
            char* w = malloc(strlen(user_msg) + 4); sprintf(w, "\n%s", user_msg);
            count += tokenize_raw(model, w, tokens + count, max_tokens - count); free(w);
            count = append_special(model, "<|eot_id|>", tokens, count, max_tokens);
            count = append_special(model, "<|start_header_id|>", tokens, count, max_tokens);
            count += tokenize_raw(model, "assistant", tokens + count, max_tokens - count);
            count = append_special(model, "<|end_header_id|>", tokens, count, max_tokens);
        }
    } else if (tmpl == 2) {
        count = append_special(model, "<|im_start|>", tokens, count, max_tokens);
        char* w = malloc(strlen(user_msg) + 16); sprintf(w, "user\n%s", user_msg);
        count += tokenize_raw(model, w, tokens + count, max_tokens - count); free(w);
        count = append_special(model, "<|im_end|>", tokens, count, max_tokens);
        count = append_special(model, "<|im_start|>", tokens, count, max_tokens);
        count += tokenize_raw(model, "assistant\n", tokens + count, max_tokens - count);
    } else if (tmpl == 3) {
        int id = find_special_token(model, "[INST]");
        int eid = find_special_token(model, "[/INST]");
        if (id >= 0) tokens[count++] = id;
        char* w = malloc(strlen(user_msg) + 4); sprintf(w, " %s ", user_msg);
        count += tokenize_raw(model, w, tokens + count, max_tokens - count); free(w);
        if (eid >= 0) tokens[count++] = eid;
    } else {
        count += tokenize_raw(model, user_msg, tokens + count, max_tokens - count);
    }
    return count;
}

/* Detokenize token IDs back to text */
static int detokenize(transformer_model_t* model, const int* tokens, int ntok, char* out, int osz) {
    int pos = 0; out[0] = '\0';
    if (!model->vocab_loaded || !model->vocab_tokens) {
        snprintf(out, osz, "[no tokenizer - %d tokens]", ntok);
        return (int)strlen(out);
    }
    for (int i = 0; i < ntok && pos < osz - 1; i++) {
        int id = tokens[i];
        if (id <= 0 || id == 1 || id == 2) continue;
        if (id >= model->config.vocab_size) continue;
        const char* tok = model->vocab_tokens[id];
        if (!tok) continue;
        const char* p = tok;
        while (*p && pos < osz - 1) {
            if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
                out[pos++] = ' '; p += 3;
            } else { out[pos++] = *p++; }
        }
    }
    out[pos] = '\0';
    return pos;
}

/* ── Sampling ─────────────────────────────────────────────────────── */

static uint32_t g_rng = 0;
static uint32_t rng_next(void) {
    if (!g_rng) g_rng = (uint32_t)time(NULL) ^ 0xDEADBEEF;
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
static float rng_float(void) { return (float)(rng_next() & 0xFFFFFF) / (float)0xFFFFFF; }

static int sample_token(float* logits, int vocab, float temperature, int top_k,
                        float rep_penalty, const int* recent, int n_recent) {
    if (rep_penalty > 1.0f && recent && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            int t = recent[i];
            if (t >= 0 && t < vocab) {
                if (logits[t] > 0) logits[t] /= rep_penalty;
                else logits[t] *= rep_penalty;
            }
        }
    }
    if (temperature < 0.01f) {
        float best = logits[0]; int bid = 0;
        for (int v = 1; v < vocab; v++) if (logits[v] > best) { best = logits[v]; bid = v; }
        return bid;
    }
    for (int v = 0; v < vocab; v++) logits[v] /= temperature;
    int k = (top_k > 0 && top_k < vocab) ? top_k : 40;
    float threshold = -1e30f;
    if (k < vocab) {
        float* topk = malloc(k * sizeof(float));
        for (int i = 0; i < k; i++) topk[i] = -1e30f;
        float hmin = -1e30f;
        for (int v = 0; v < vocab; v++) {
            if (logits[v] > hmin) {
                int mi = 0; for (int j = 1; j < k; j++) if (topk[j] < topk[mi]) mi = j;
                topk[mi] = logits[v];
                hmin = topk[0]; for (int j = 1; j < k; j++) if (topk[j] < hmin) hmin = topk[j];
            }
        }
        threshold = hmin; free(topk);
    }
    float mx = -1e30f;
    for (int v = 0; v < vocab; v++) if (logits[v] >= threshold && logits[v] > mx) mx = logits[v];
    float sum = 0;
    for (int v = 0; v < vocab; v++) {
        if (logits[v] >= threshold) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
        else logits[v] = 0;
    }
    if (sum <= 0) { float best = logits[0]; int bid = 0; for (int v = 1; v < vocab; v++) if (logits[v] > best) { best = logits[v]; bid = v; } return bid; }
    for (int v = 0; v < vocab; v++) logits[v] /= sum;
    float r = rng_float(), cs = 0;
    for (int v = 0; v < vocab; v++) { cs += logits[v]; if (cs >= r) return v; }
    return vocab - 1;
}

/* ── Main ─────────────────────────────────────────────────────────── */

void print_banner(void) {
    printf("\n========================================\n");
    printf("       FAST LLM - REAL AI CHAT         \n");
    printf("========================================\n\n");
    printf("Commands: /help /reset /stats /layers N /quit\n\n");
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) { printf("Usage: chat <model.gguf>\n"); return 1; }

    printf("Loading model: %s\n", argv[1]);
    transformer_model_t* model = model_load_gguf(argv[1], 1);
    if (!model) { printf("ERROR: Failed to load model!\n"); return 1; }

    model_print_info(model);
    g_max_layers = model->config.num_layers;

    float temperature = 0.7f;
    int top_k = 40;
    float rep_penalty = 1.1f;
    int max_gen = 256;

    /* Detect EOS tokens */
    int eos_ids[8] = {2, -1, -1, -1, -1, -1, -1, -1};
    int n_eos = 1;
    const char* eos_names[] = {"<|end|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_text|>", NULL};
    for (int i = 0; eos_names[i] && n_eos < 8; i++) {
        int id = find_special_token(model, eos_names[i]);
        if (id >= 0) eos_ids[n_eos++] = id;
    }

    printf("Ready! Type your message.\n\n");

    char input[1024];
    while (1) {
        printf("You: "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (!input[0]) continue;

        if (input[0] == '/') {
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) { printf("\nGoodbye!\n"); break; }
            if (strcmp(input, "/help") == 0) { print_banner(); continue; }
            if (strcmp(input, "/stats") == 0) { model_print_info(model); continue; }
            if (strncmp(input, "/layers ", 8) == 0) {
                int l = atoi(input + 8);
                if (l >= 1 && l <= model->config.num_layers) { g_max_layers = l; printf("Using %d layers.\n\n", l); }
                else printf("Invalid (1-%d).\n\n", model->config.num_layers);
                continue;
            }
            if (strcmp(input, "/reset") == 0) { printf("Conversation reset.\n\n"); continue; }
            printf("Unknown command.\n\n"); continue;
        }

        /* Tokenize with chat template */
        int ctx[4096];
        int ctx_len = tokenize_chat(model, input, ctx, 1024);

        int vocab = model->config.vocab_size;
        float* logits = aligned_malloc(vocab * sizeof(float), 64);
        int out_tokens[512];
        int gen = 0;

        printf("Assistant: "); fflush(stdout);

        for (int t = 0; t < max_gen; t++) {
            int next = 0;
            model_forward(model, ctx, ctx_len, logits, &next);
            next = sample_token(logits, vocab, temperature, top_k, rep_penalty, out_tokens, gen);

            if (ctx_len < 4096) ctx[ctx_len++] = next;
            else { memmove(ctx, ctx + 64, (ctx_len - 64) * sizeof(int)); ctx_len -= 64; ctx[ctx_len++] = next; }

            out_tokens[gen++] = next;

            /* Print token as it's generated (streaming) */
            if (model->vocab_loaded && model->vocab_tokens && next > 2 && next < vocab) {
                const char* tok = model->vocab_tokens[next];
                if (tok) {
                    const char* p = tok;
                    while (*p) {
                        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
                            putchar(' '); p += 3;
                        } else { putchar(*p); p++; }
                    }
                    fflush(stdout);
                }
            }

            /* Check EOS */
            int is_eos = 0;
            for (int e = 0; e < n_eos; e++) if (next == eos_ids[e]) { is_eos = 1; break; }
            if (is_eos) break;
            if (gen >= 5) {
                int same = 1;
                for (int r = 1; r < 5; r++) if (out_tokens[gen-1-r] != out_tokens[gen-1]) { same = 0; break; }
                if (same) break;
            }
        }

        printf("\n\n");
        aligned_free(logits);
    }

    model_free(model);
    return 0;
}
