/*
 * fllm CLI - Daemon (background server), client connection
 */
#include "cli.h"
#include "model_loader.h"

/* External inference */
extern void model_forward(transformer_model_t* model,
                          const int* input_tokens, int seq_len,
                          float* output_logits, int* output_tokens);
extern int g_max_layers;

static volatile int g_running = 1;
static transformer_model_t* g_model = NULL;

/* ── Network init ─────────────────────────────────────────────────── */

void net_init(void) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

void net_cleanup(void) {
#ifdef _WIN32
    WSACleanup();
#endif
}

/* ── PID file ─────────────────────────────────────────────────────── */

void daemon_write_pid(void) {
    char path[1024];
    cli_get_pid_path(path, sizeof(path));
    FILE* f = fopen(path, "w");
    if (f) { fprintf(f, "%d\n%d\n", GETPID(), DAEMON_PORT); fclose(f); }
}

void daemon_remove_pid(void) {
    char path[1024];
    cli_get_pid_path(path, sizeof(path));
    remove(path);
}

int daemon_read_pid(int* pid, int* port) {
    char path[1024];
    cli_get_pid_path(path, sizeof(path));
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int p = 0, pt = 0;
    if (fscanf(f, "%d\n%d", &p, &pt) < 1) { fclose(f); return 0; }
    fclose(f);
    if (pid) *pid = p;
    if (port) *port = pt ? pt : DAEMON_PORT;
    return p > 0;
}

static int process_alive(int pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return 0;
    DWORD code;
    int alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    return kill(pid, 0) == 0;
#endif
}

int daemon_is_running(int* out_port) {
    int pid, port;
    if (!daemon_read_pid(&pid, &port)) return 0;
    if (!process_alive(pid)) { daemon_remove_pid(); return 0; }
    if (out_port) *out_port = port;
    return 1;
}

/* ── Send command to daemon ───────────────────────────────────────── */

int daemon_send(int port, const char* cmd, char* resp, int rsz) {
    net_init();
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == SOCK_INVALID) { net_cleanup(); return 0; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        sock_close(s); net_cleanup(); return 0;
    }

    send(s, cmd, (int)strlen(cmd), 0);

    /* Set receive timeout to 60 seconds */
#ifdef _WIN32
    DWORD tv_ms = 60000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv_ms, sizeof(tv_ms));
#else
    struct timeval tv_so = { 60, 0 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv_so, sizeof(tv_so));
#endif

    int total = 0;
    while (total < rsz - 1) {
        int n = recv(s, resp + total, rsz - 1 - total, 0);
        if (n <= 0) break;
        total += n;
        resp[total] = '\0';
        if (strstr(resp, PROTO_END)) break;
    }
    char* end = strstr(resp, PROTO_END);
    if (end) *end = '\0';

    sock_close(s); net_cleanup();
    return 1;
}

/* ── Tokenizer using model's real vocabulary ───────────────────────── */

/* SentencePiece "▁" in UTF-8: 0xE2 0x96 0x81 */
#define SP_SPACE "\xE2\x96\x81"
#define SP_SPACE_LEN 3

/* BPE "Ġ" in UTF-8: 0xC4 0xA0 (used by Llama 3, GPT-2 style tokenizers) */
#define BPE_SPACE "\xC4\xA0"
#define BPE_SPACE_LEN 2

/*
 * Detect tokenizer convention by scanning vocabulary.
 * Returns: 1 = SentencePiece (▁), 2 = BPE (Ġ), 0 = unknown/plain
 */
static int detect_tokenizer_type(transformer_model_t* model) {
    if (!model->vocab_loaded || !model->vocab_tokens) return 0;
    int sp_count = 0, bpe_count = 0;
    int limit = model->config.vocab_size < 10000 ? model->config.vocab_size : 10000;
    for (int v = 0; v < limit; v++) {
        const char* tok = model->vocab_tokens[v];
        if (!tok) continue;
        if ((unsigned char)tok[0] == 0xE2 && (unsigned char)tok[1] == 0x96 && (unsigned char)tok[2] == 0x81)
            sp_count++;
        if ((unsigned char)tok[0] == 0xC4 && (unsigned char)tok[1] == 0xA0)
            bpe_count++;
    }
    if (sp_count > bpe_count && sp_count > 50) return 1;  /* SentencePiece */
    if (bpe_count > sp_count && bpe_count > 50) return 2;  /* BPE */
    return 0;  /* Plain / no space prefix */
}

/*
 * Find a special token by exact string match in the vocabulary.
 * Returns token ID or -1 if not found.
 */
static int find_special_token(transformer_model_t* model, const char* text) {
    if (!model->vocab_loaded || !model->vocab_tokens) return -1;
    for (int v = 0; v < model->config.vocab_size; v++) {
        const char* tok = model->vocab_tokens[v];
        if (tok && strcmp(tok, text) == 0) return v;
    }
    return -1;
}

/*
 * Tokenize a special token string (e.g. "<|user|>") and append its ID.
 * If not found in vocab, silently skip.
 */
static int append_special(transformer_model_t* model, const char* tag, int* tokens, int count, int max_tokens) {
    if (count >= max_tokens) return count;
    int id = find_special_token(model, tag);
    if (id >= 0) tokens[count++] = id;
    return count;
}

/*
 * BPE-style tokenize: greedy longest match against vocab.
 * Handles both SentencePiece (▁) and BPE (Ġ) space conventions.
 * For plain tokenizers (no space prefix), matches raw text directly.
 */
static int real_tokenize_raw(transformer_model_t* model, const char* text, int* tokens, int max_tokens) {
    int count = 0;

    if (!model->vocab_loaded || !model->vocab_tokens) {
        /* Fallback: byte-level encoding using low token IDs */
        const unsigned char* p = (const unsigned char*)text;
        while (*p && count < max_tokens - 1) {
            tokens[count++] = 3 + *p;
            p++;
        }
        return count;
    }

    int tok_type = detect_tokenizer_type(model);
    const char* space_prefix = NULL;
    int space_prefix_len = 0;

    if (tok_type == 1) {
        space_prefix = SP_SPACE;
        space_prefix_len = SP_SPACE_LEN;
    } else if (tok_type == 2) {
        space_prefix = BPE_SPACE;
        space_prefix_len = BPE_SPACE_LEN;
    }

    /* Normalize text: convert spaces to the tokenizer's space prefix */
    char* normalized = NULL;
    const char* match_text = text;

    if (space_prefix) {
        int norm_sz = (int)strlen(text) * 4 + 16;
        normalized = malloc(norm_sz);
        int pos = 0;
        const char* p = text;

        /* Prepend space prefix to first word (standard for both SP and BPE) */
        if (*p && *p != ' ') {
            memcpy(normalized + pos, space_prefix, space_prefix_len);
            pos += space_prefix_len;
        }

        while (*p && pos < norm_sz - 4) {
            if (*p == ' ') {
                memcpy(normalized + pos, space_prefix, space_prefix_len);
                pos += space_prefix_len;
                p++;
            } else {
                normalized[pos++] = *p++;
            }
        }
        normalized[pos] = '\0';
        match_text = normalized;
    }

    int vocab_size = model->config.vocab_size;
    const char* p = match_text;

    while (*p && count < max_tokens - 1) {
        int best_len = 0;
        int best_id = -1;

        /* Greedy longest-match against vocabulary */
        for (int v = 0; v < vocab_size; v++) {
            const char* tok = model->vocab_tokens[v];
            if (!tok || tok[0] == '\0') continue;

            /* Quick first-byte filter */
            if ((unsigned char)tok[0] != (unsigned char)*p) continue;

            int tlen = (int)strlen(tok);
            if (tlen > best_len && strncmp(p, tok, tlen) == 0) {
                best_len = tlen;
                best_id = v;
            }
        }

        if (best_len > 0 && best_id >= 0) {
            tokens[count++] = best_id;
            p += best_len;
        } else {
            /* Single byte fallback — try to find a byte token */
            int found = 0;
            unsigned char ch = (unsigned char)*p;
            for (int v = 0; v < vocab_size; v++) {
                const char* tok = model->vocab_tokens[v];
                if (tok && strlen(tok) == 1 && (unsigned char)tok[0] == ch) {
                    tokens[count++] = v;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                /* Last resort: use byte value + 3 as token ID */
                tokens[count++] = 3 + ch;
            }
            p++;
        }
    }

    if (normalized) free(normalized);
    return count;
}

/*
 * Detect chat template style from vocabulary.
 * Returns: 1=Llama3/Phi3 style, 2=ChatML style, 3=Llama2 style, 0=unknown
 */
static int detect_template(transformer_model_t* model) {
    if (!model->vocab_loaded || !model->vocab_tokens) return 0;

    /* Check for Llama 3 / Phi-3 style tokens */
    if (find_special_token(model, "<|user|>") >= 0 ||
        find_special_token(model, "<|start_header_id|>") >= 0)
        return 1;

    /* Check for ChatML style */
    if (find_special_token(model, "<|im_start|>") >= 0)
        return 2;

    /* Check for Llama 2 style */
    if (find_special_token(model, "[INST]") >= 0)
        return 3;

    return 0;
}

/*
 * Full tokenization with chat template wrapping.
 * Produces: BOS + template_header + user_message + template_footer
 */
static int real_tokenize(transformer_model_t* model, const char* text, int* tokens, int max_tokens) {
    int count = 0;
    tokens[count++] = 1; /* BOS */

    int tmpl = detect_template(model);
    fprintf(stderr, "[tokenize] template=%d\n", tmpl);

    if (tmpl == 1) {
        /* Llama 3 / Phi-3 style:
         * <|user|>\n{message}<|end|>\n<|assistant|>\n
         * OR: <|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
         */
        if (find_special_token(model, "<|user|>") >= 0) {
            /* Phi-3 style */
            count = append_special(model, "<|user|>", tokens, count, max_tokens);
            /* Tokenize "\n" + message */
            char* wrapped = malloc(strlen(text) + 4);
            sprintf(wrapped, "\n%s", text);
            count += real_tokenize_raw(model, wrapped, tokens + count, max_tokens - count);
            free(wrapped);
            count = append_special(model, "<|end|>", tokens, count, max_tokens);
            count = append_special(model, "<|assistant|>", tokens, count, max_tokens);
        } else {
            /* Llama 3 style */
            count = append_special(model, "<|start_header_id|>", tokens, count, max_tokens);
            count += real_tokenize_raw(model, "user", tokens + count, max_tokens - count);
            count = append_special(model, "<|end_header_id|>", tokens, count, max_tokens);
            char* wrapped = malloc(strlen(text) + 4);
            sprintf(wrapped, "\n%s", text);
            count += real_tokenize_raw(model, wrapped, tokens + count, max_tokens - count);
            free(wrapped);
            count = append_special(model, "<|eot_id|>", tokens, count, max_tokens);
            count = append_special(model, "<|start_header_id|>", tokens, count, max_tokens);
            count += real_tokenize_raw(model, "assistant", tokens + count, max_tokens - count);
            count = append_special(model, "<|end_header_id|>", tokens, count, max_tokens);
        }
    } else if (tmpl == 2) {
        /* ChatML style:
         * <|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n
         */
        count = append_special(model, "<|im_start|>", tokens, count, max_tokens);
        char* wrapped = malloc(strlen(text) + 16);
        sprintf(wrapped, "user\n%s", text);
        count += real_tokenize_raw(model, wrapped, tokens + count, max_tokens - count);
        free(wrapped);
        count = append_special(model, "<|im_end|>", tokens, count, max_tokens);
        count = append_special(model, "<|im_start|>", tokens, count, max_tokens);
        count += real_tokenize_raw(model, "assistant\n", tokens + count, max_tokens - count);
    } else if (tmpl == 3) {
        /* Llama 2 style:
         * [INST] {message} [/INST]
         */
        int inst_id = find_special_token(model, "[INST]");
        int inst_end_id = find_special_token(model, "[/INST]");
        if (inst_id >= 0) tokens[count++] = inst_id;
        char* wrapped = malloc(strlen(text) + 4);
        sprintf(wrapped, " %s ", text);
        count += real_tokenize_raw(model, wrapped, tokens + count, max_tokens - count);
        free(wrapped);
        if (inst_end_id >= 0) tokens[count++] = inst_end_id;
    } else {
        /* No template detected — just tokenize the raw text */
        count += real_tokenize_raw(model, text, tokens + count, max_tokens - count);
    }

    return count;
}

/* Detokenize: convert token IDs back to text */
static int real_detokenize(transformer_model_t* model, const int* tokens, int ntok, char* out, int osz) {
    int pos = 0;
    out[0] = '\0';

    if (!model->vocab_loaded || !model->vocab_tokens) {
        snprintf(out, osz, "[no tokenizer - %d tokens generated]", ntok);
        return (int)strlen(out);
    }

    for (int i = 0; i < ntok && pos < osz - 1; i++) {
        int id = tokens[i];
        if (id <= 0 || id == 1 || id == 2) continue; /* skip PAD, BOS, EOS */
        if (id >= model->config.vocab_size) continue;

        const char* tok = model->vocab_tokens[id];
        if (!tok) continue;

        /* Handle both space conventions:
         * SentencePiece: "▁" (UTF-8: 0xE2 0x96 0x81) → space
         * BPE:           "Ġ" (UTF-8: 0xC4 0xA0) → space
         */
        const char* p = tok;
        while (*p && pos < osz - 1) {
            if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
                out[pos++] = ' ';
                p += 3;
            } else if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0xA0) {
                out[pos++] = ' ';
                p += 2;
            } else {
                out[pos++] = *p++;
            }
        }
    }
    out[pos] = '\0';
    return pos;
}

/* ── Sampling with temperature and top-k ───────────────────────────── */

/* Simple xorshift RNG (no dependency on srand/rand state) */
static uint32_t g_rng_state = 0;
static uint32_t rng_next(void) {
    if (g_rng_state == 0) g_rng_state = (uint32_t)time(NULL) ^ 0xDEADBEEF;
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    return g_rng_state;
}
static float rng_float(void) { return (float)(rng_next() & 0xFFFFFF) / (float)0xFFFFFF; }

/*
 * Sample a token from logits with temperature, top-k, and repetition penalty.
 *   temperature: 0 = greedy, >0 = random sampling
 *   top_k: number of top candidates to sample from (0 = disabled)
 *   rep_penalty: >1.0 penalizes recently generated tokens
 *   recent_tokens/n_recent: window of recent tokens for repetition penalty
 */
static int sample_with_params(float* logits, int vocab,
                              float temperature, int top_k,
                              float rep_penalty,
                              const int* recent_tokens, int n_recent) {
    /* Apply repetition penalty */
    if (rep_penalty > 1.0f && recent_tokens && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            int tid = recent_tokens[i];
            if (tid >= 0 && tid < vocab) {
                if (logits[tid] > 0)
                    logits[tid] /= rep_penalty;
                else
                    logits[tid] *= rep_penalty;
            }
        }
    }

    /* Greedy if temperature is 0 or very small */
    if (temperature < 0.01f) {
        float best = logits[0]; int bid = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > best) { best = logits[v]; bid = v; }
        return bid;
    }

    /* Apply temperature */
    for (int v = 0; v < vocab; v++)
        logits[v] /= temperature;

    /* Top-K filtering: find top_k largest logits */
    int k = (top_k > 0 && top_k < vocab) ? top_k : 40;
    /* Simple partial sort: find the k-th largest value */
    /* For efficiency, use a single pass to find the threshold */
    float threshold = -1e30f;
    if (k < vocab) {
        /* Collect top-k values using a simple min-heap approach */
        float* topk = (float*)malloc(k * sizeof(float));
        for (int i = 0; i < k; i++) topk[i] = -1e30f;
        float heap_min = -1e30f;
        for (int v = 0; v < vocab; v++) {
            if (logits[v] > heap_min) {
                /* Find and replace the minimum in topk */
                int min_idx = 0;
                for (int j = 1; j < k; j++)
                    if (topk[j] < topk[min_idx]) min_idx = j;
                topk[min_idx] = logits[v];
                /* Update heap_min */
                heap_min = topk[0];
                for (int j = 1; j < k; j++)
                    if (topk[j] < heap_min) heap_min = topk[j];
            }
        }
        threshold = heap_min;
        free(topk);
    }

    /* Softmax over candidates above threshold */
    float max_val = -1e30f;
    for (int v = 0; v < vocab; v++) {
        if (logits[v] >= threshold && logits[v] > max_val)
            max_val = logits[v];
    }

    float sum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        if (logits[v] >= threshold) {
            logits[v] = expf(logits[v] - max_val);
            sum += logits[v];
        } else {
            logits[v] = 0.0f;
        }
    }

    if (sum <= 0.0f) {
        /* Fallback to greedy */
        float best = logits[0]; int bid = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > best) { best = logits[v]; bid = v; }
        return bid;
    }

    /* Normalize */
    for (int v = 0; v < vocab; v++)
        logits[v] /= sum;

    /* Random sample */
    float r = rng_float();
    float cumsum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        cumsum += logits[v];
        if (cumsum >= r) return v;
    }
    return vocab - 1; /* shouldn't reach here */
}

/* ── Generate with real tokenizer ─────────────────────────────────── */

/* Global chat parameters (set from handle_cmd) */
static float g_temperature = 0.7f;
static int   g_top_k = 40;
static float g_rep_penalty = 1.1f;

static void run_chat(transformer_model_t* model, const char* user_msg, int ntok, char* out, int osz) {
    int vocab = model->config.vocab_size;
    fprintf(stderr, "[run_chat] vocab=%d msg='%s' ntok=%d temp=%.2f\n", vocab, user_msg, ntok, g_temperature);
    fflush(stderr);

    float* logits = (float*)aligned_malloc(vocab * sizeof(float), 64);
    int* ctx = (int*)malloc(4096 * sizeof(int));
    int ctx_len = 0;

    /* Tokenize user message with chat template */
    fprintf(stderr, "[run_chat] tokenizing...\n"); fflush(stderr);
    ctx_len = real_tokenize(model, user_msg, ctx, 1024);
    fprintf(stderr, "[run_chat] tokenized: %d tokens, first few:", ctx_len);
    for (int i = 0; i < ctx_len && i < 10; i++) fprintf(stderr, " %d", ctx[i]);
    fprintf(stderr, "\n"); fflush(stderr);

    double start = cli_time_sec();
    int gen = 0;
    int out_tokens[512];

    /* Detect EOS tokens for this model */
    int eos_ids[8] = {2, -1, -1, -1, -1, -1, -1, -1};
    int n_eos = 1;
    /* Common EOS/stop tokens */
    const char* eos_names[] = {
        "<|end|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>",
        "</s>", "<|end_of_text|>", NULL
    };
    for (int i = 0; eos_names[i] && n_eos < 8; i++) {
        int id = find_special_token(model, eos_names[i]);
        if (id >= 0) eos_ids[n_eos++] = id;
    }

    for (int t = 0; t < ntok; t++) {
        int next = 0;
        fprintf(stderr, "[run_chat] forward pass %d/%d ctx_len=%d...\n", t+1, ntok, ctx_len); fflush(stderr);
        model_forward(model, ctx, ctx_len, logits, &next);

        /* Sample with temperature, top-k, and repetition penalty */
        next = sample_with_params(logits, vocab,
                                  g_temperature, g_top_k, g_rep_penalty,
                                  out_tokens, gen);

        if (ctx_len < 4096) ctx[ctx_len++] = next;
        else { memmove(ctx, ctx+64, (ctx_len-64)*sizeof(int)); ctx_len -= 64; ctx[ctx_len++] = next; }

        out_tokens[gen] = next;
        gen++;

        /* Stop on any EOS token */
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next == eos_ids[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;

        /* Stop if stuck repeating the same token 5 times */
        if (gen >= 5) {
            int all_same = 1;
            for (int r = 1; r < 5; r++)
                if (out_tokens[gen-1-r] != out_tokens[gen-1]) { all_same = 0; break; }
            if (all_same) break;
        }
    }

    double elapsed = cli_time_sec() - start;
    double tps = (elapsed > 0.001) ? gen / elapsed : 0;
    fprintf(stderr, "[run_chat] generated %d tokens in %.2fs, first token=%d\n", gen, elapsed, gen > 0 ? out_tokens[0] : -1);
    fflush(stderr);

    /* Detokenize the generated tokens */
    char text_out[4096] = {0};
    real_detokenize(model, out_tokens, gen, text_out, sizeof(text_out));

    int pos = 0;
    if (text_out[0]) {
        pos += snprintf(out + pos, osz - pos, "%s", text_out);
    } else if (gen > 0) {
        /* Show raw token IDs if detokenizer produced nothing */
        pos += snprintf(out + pos, osz - pos, "[tokens:");
        for (int i = 0; i < gen && i < 20; i++)
            pos += snprintf(out + pos, osz - pos, " %d", out_tokens[i]);
        if (gen > 20) pos += snprintf(out + pos, osz - pos, " ...");
        pos += snprintf(out + pos, osz - pos, "]");
    }
    pos += snprintf(out + pos, osz - pos,
        "\n  [%d tokens | %.1f tok/s | %.0fms]",
        gen, tps, elapsed * 1000.0);

    free(ctx); aligned_free(logits);
}

static void run_generate(transformer_model_t* model, int ntok, char* out, int osz) {
    run_chat(model, "The meaning of life is", ntok, out, osz);
}

/* ── Handle daemon command ────────────────────────────────────────── */

static void handle_cmd(const char* raw, char* resp, int rsz) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "%s", raw);
    /* Trim */
    while (cmd[0] == ' ') memmove(cmd, cmd+1, strlen(cmd));
    int len = (int)strlen(cmd);
    while (len > 0 && (cmd[len-1] == '\n' || cmd[len-1] == '\r' || cmd[len-1] == ' '))
        cmd[--len] = '\0';

    if (strcmp(cmd, "status") == 0) {
        if (g_model)
            snprintf(resp, rsz, "Daemon PID %d\nModel: %s\n  %d layers, hidden %d, vocab %d, Q%d\n",
                GETPID(), g_model->config.model_name, g_model->config.num_layers,
                g_model->config.hidden_size, g_model->config.vocab_size, g_model->config.quant_bits);
        else
            snprintf(resp, rsz, "Daemon PID %d\nNo model loaded.\n", GETPID());
    }
    else if (strncmp(cmd, "gen", 3) == 0) {
        if (!g_model) { snprintf(resp, rsz, "No model loaded.\n"); return; }
        int n = 50;
        char* sp = strchr(cmd, ' ');
        if (sp) n = atoi(sp+1);
        if (n < 1)
            n = 50;
        if (n > 500)
            n = 500;
        run_generate(g_model, n, resp, rsz);
    }
    else if (strncmp(cmd, "chat ", 5) == 0) {
        if (!g_model) { snprintf(resp, rsz, "No model loaded.\n"); return; }
        /* Format: "chat <max_tokens> <user message>"
         * Or:     "chat <max_tokens> <temp> <top_k> <rep_penalty> <user message>"
         */
        const char* rest = cmd + 5;
        int n = 256;
        char* endp = NULL;
        long val = strtol(rest, &endp, 10);
        const char* msg = rest;
        if (endp && endp != rest && *endp == ' ') {
            n = (int)val;
            msg = endp + 1;

            /* Try to parse optional temp, top_k, rep_penalty */
            float temp = 0;
            long tk = 0;
            float rp = 0;
            char* endp2 = NULL;
            temp = strtof(msg, &endp2);
            if (endp2 && endp2 != msg && *endp2 == ' ') {
                msg = endp2 + 1;
                char* endp3 = NULL;
                tk = strtol(msg, &endp3, 10);
                if (endp3 && endp3 != msg && *endp3 == ' ') {
                    msg = endp3 + 1;
                    char* endp4 = NULL;
                    rp = strtof(msg, &endp4);
                    if (endp4 && endp4 != msg && *endp4 == ' ') {
                        msg = endp4 + 1;
                        g_temperature = temp;
                        g_top_k = (int)tk;
                        g_rep_penalty = rp;
                    }
                }
            }
        }
        if (n < 1)
            n = 1;
        if (n > 500)
            n = 500;
        run_chat(g_model, msg, n, resp, rsz);
    }
    else if (strcmp(cmd, "bench") == 0) {
        if (!g_model) { snprintf(resp, rsz, "No model loaded.\n"); return; }
        run_generate(g_model, 10, resp, rsz);
    }
    else if (strcmp(cmd, "shutdown") == 0 || strcmp(cmd, "off") == 0) {
        snprintf(resp, rsz, "Shutting down daemon...\n");
        g_running = 0;
    }
    else if (strcmp(cmd, "help") == 0) {
        snprintf(resp, rsz,
            "Commands:\n"
            "  status       Daemon & model info\n"
            "  generate N   Generate N tokens (default 50)\n"
            "  bench        Quick 10-token test\n"
            "  off          Shutdown daemon\n"
            "  help         This message\n");
    }
    else {
        snprintf(resp, rsz, "Unknown: '%s'. Type 'help'.\n", cmd);
    }
}

/* ── Daemon server loop ───────────────────────────────────────────── */

void daemon_serve(const char* model_path) {
    cli_ensure_dirs();

    if (model_path && model_path[0]) {
        fprintf(stdout, "  Loading model: %s\n", model_path);
        fflush(stdout);
        g_model = model_load_gguf(model_path, 1);
        if (g_model) {
            g_max_layers = g_model->config.num_layers;
            fprintf(stdout, "  Loaded: %s (%d layers)\n", g_model->config.model_name, g_model->config.num_layers);
        } else {
            fprintf(stdout, "  Warning: Failed to load model.\n");
        }
        fflush(stdout);
    }

    net_init();
    sock_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == SOCK_INVALID) { fprintf(stdout, "  Error: socket failed.\n"); return; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(DAEMON_PORT);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stdout, "  Error: port %d in use.\n", DAEMON_PORT);
        sock_close(srv); net_cleanup(); return;
    }

    listen(srv, 4);
    daemon_write_pid();
    fprintf(stdout, "  Daemon on 127.0.0.1:%d (PID %d)\n", DAEMON_PORT, GETPID());
    fflush(stdout);

    while (g_running) {
        fd_set fds; FD_ZERO(&fds); FD_SET(srv, &fds);
        struct timeval tv = { 1, 0 };
#ifdef _WIN32
        if (select(0, &fds, NULL, NULL, &tv) <= 0) continue;
#else
        if (select(srv + 1, &fds, NULL, NULL, &tv) <= 0) continue;
#endif
        sock_t cl = accept(srv, NULL, NULL);
        if (cl == SOCK_INVALID) continue;

        char buf[4096] = {0};
        int n = recv(cl, buf, sizeof(buf)-1, 0);
        if (n > 0) {
            buf[n] = '\0';
            char resp[4096] = {0};
            handle_cmd(buf, resp, sizeof(resp));
            send(cl, resp, (int)strlen(resp), 0);
            send(cl, PROTO_END, (int)strlen(PROTO_END), 0);
        }
        sock_close(cl);
    }

    sock_close(srv);
    daemon_remove_pid();
    if (g_model) { model_free(g_model); g_model = NULL; }
    net_cleanup();
    fprintf(stdout, "  Daemon stopped.\n");
}

/* ── Daemonize (detach) ───────────────────────────────────────────── */

void daemon_start_detached(const char* model_path) {
#ifdef _WIN32
    char exe[1024], cmdline[2048];
    GetModuleFileNameA(NULL, exe, sizeof(exe));
    if (model_path && model_path[0])
        snprintf(cmdline, sizeof(cmdline), "\"%s\" --_daemon --model \"%s\"", exe, model_path);
    else
        snprintf(cmdline, sizeof(cmdline), "\"%s\" --_daemon", exe);

    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si)); si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));
    if (CreateProcessA(NULL, cmdline, NULL, NULL, FALSE,
                       DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                       NULL, NULL, &si, &pi)) {
        printf("  Daemon started (PID %lu)\n", pi.dwProcessId);
        CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    } else {
        printf("  Error: Failed to start daemon.\n");
    }
#else
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return; }
    if (pid > 0) { printf("  Daemon started (PID %d)\n", pid); return; }
    setsid();
    char logpath[1024];
    cli_get_log_path(logpath, sizeof(logpath));
    FILE* lf = freopen(logpath, "a", stdout);
    if (lf) freopen(logpath, "a", stderr);
    (void)lf;
    daemon_serve(model_path);
    exit(0);
#endif
}

/* ── Interactive client ───────────────────────────────────────────── */

void daemon_interactive(int port) {
    printf("\n  Connected to daemon (port %d)\n", port);
    printf("  Type 'help' for commands, 'off' to shutdown.\n\n");

    char resp[4096];
    if (daemon_send(port, "status", resp, sizeof(resp)))
        printf("  %s\n", resp);

    while (1) {
        printf("  fllm> ");
        fflush(stdout);
        char input[256];
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = '\0';
        if (strlen(input) == 0) continue;

        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("  Disconnected. Daemon still running. Use 'fllm -off' to stop.\n\n");
            break;
        }

        memset(resp, 0, sizeof(resp));
        if (daemon_send(port, input, resp, sizeof(resp))) {
            printf("  %s\n", resp);
            if (strstr(resp, "Shutting down")) break;
        } else {
            printf("  Lost connection.\n"); break;
        }
    }
}

/* ── Stop daemon ──────────────────────────────────────────────────── */

void daemon_stop(void) {
    int port;
    if (!daemon_is_running(&port)) {
        printf("  No daemon running.\n\n");
        return;
    }
    char resp[4096];
    if (daemon_send(port, "shutdown", resp, sizeof(resp)))
        printf("  %s", resp);
    else
        printf("  Could not connect. Cleaning up.\n");
    daemon_remove_pid();
}
