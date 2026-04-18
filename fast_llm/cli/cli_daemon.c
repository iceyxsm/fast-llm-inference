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

/* BPE-style tokenize: greedy longest match against vocab */
static int real_tokenize(transformer_model_t* model, const char* text, int* tokens, int max_tokens) {
    int count = 0;
    tokens[count++] = 1; /* BOS */

    if (!model->vocab_loaded || !model->vocab_tokens) {
        /* Fallback: hash-based */
        const char* p = text;
        while (*p && count < max_tokens - 1) {
            while (*p == ' ' || *p == '\t' || *p == '\n') p++;
            if (!*p) break;
            const char* start = p;
            while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
            unsigned int hash = 5381;
            for (const char* c = start; c < p; c++)
                hash = ((hash << 5) + hash) + (unsigned char)*c;
            tokens[count++] = 100 + (hash % (model->config.vocab_size - 200));
        }
        if (count < max_tokens) tokens[count++] = 2;
        return count;
    }

    /* Real tokenization: greedy longest-match against vocab
       Optimization: only check tokens that start with the current character */
    int vocab_size = model->config.vocab_size;
    const char* p = text;

    while (*p && count < max_tokens - 1) {
        int best_len = 0;
        int best_id = 3; /* unknown token */
        unsigned char first_ch = (unsigned char)*p;

        /* Only check vocab entries that start with the same byte */
        for (int v = 0; v < vocab_size; v++) {
            const char* tok = model->vocab_tokens[v];
            if (!tok || tok[0] == '\0') continue;
            if ((unsigned char)tok[0] != first_ch) {
                /* Also check for ▁ prefix (0xE2) matching space */
                if (first_ch == ' ' && (unsigned char)tok[0] == 0xE2) { /* might be ▁ */ }
                else continue;
            }
            int tlen = (int)strlen(tok);
            if (tlen > best_len && strncmp(p, tok, tlen) == 0) {
                best_len = tlen;
                best_id = v;
            }
        }

        if (best_len > 0) {
            tokens[count++] = best_id;
            p += best_len;
        } else {
            /* Single byte fallback — find the byte token */
            for (int v = 0; v < vocab_size && v < 512; v++) {
                const char* tok = model->vocab_tokens[v];
                if (tok && strlen(tok) == 1 && tok[0] == *p) { tokens[count++] = v; break; }
            }
            p++;
        }
    }

    if (count < max_tokens) tokens[count++] = 2; /* EOS */
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

        /* BPE tokens often have "▁" (U+2581) for space — replace with actual space */
        const char* p = tok;
        while (*p && pos < osz - 1) {
            /* Check for ▁ (UTF-8: 0xE2 0x96 0x81) */
            if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
                out[pos++] = ' ';
                p += 3;
            } else {
                out[pos++] = *p++;
            }
        }
    }
    out[pos] = '\0';
    return pos;
}

/* ── Generate with real tokenizer ─────────────────────────────────── */

static void run_chat(transformer_model_t* model, const char* user_msg, int ntok, char* out, int osz) {
    int vocab = model->config.vocab_size;
    fprintf(stderr, "[run_chat] vocab=%d msg='%s' ntok=%d\n", vocab, user_msg, ntok);
    fflush(stderr);

    float* logits = (float*)aligned_malloc(vocab * sizeof(float), 64);
    int* ctx = (int*)malloc(2048 * sizeof(int));
    int ctx_len = 0;

    /* Tokenize user message with real vocab */
    fprintf(stderr, "[run_chat] tokenizing...\n"); fflush(stderr);
    ctx_len = real_tokenize(model, user_msg, ctx, 512);
    fprintf(stderr, "[run_chat] tokenized: %d tokens\n", ctx_len); fflush(stderr);

    double start = cli_time_sec();
    int gen = 0;
    int out_tokens[512];

    for (int t = 0; t < ntok; t++) {
        int next = 0;
        fprintf(stderr, "[run_chat] forward pass %d/%d ctx_len=%d...\n", t+1, ntok, ctx_len); fflush(stderr);
        model_forward(model, ctx, ctx_len, logits, &next);

        /* Greedy argmax */
        float best = logits[0]; int bid = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > best) { best = logits[v]; bid = v; }
        next = bid;

        if (ctx_len < 2048) ctx[ctx_len++] = next;
        else { memmove(ctx, ctx+64, (ctx_len-64)*sizeof(int)); ctx_len -= 64; ctx[ctx_len++] = next; }

        out_tokens[gen] = next;
        gen++;
        /* Stop on EOS (2), or if model is stuck generating same token */
        if (next == 2) break;
        if (gen >= 3 && out_tokens[gen-1] == out_tokens[gen-2] && out_tokens[gen-2] == out_tokens[gen-3]) break;
    }

    double elapsed = cli_time_sec() - start;
    double tps = (elapsed > 0.001) ? gen / elapsed : 0;
    fprintf(stderr, "[run_chat] generated %d tokens in %.2fs, first token=%d\n", gen, elapsed, gen > 0 ? out_tokens[0] : -1);
    fflush(stderr);

    /* Detokenize the generated tokens */
    char text_out[2048] = {0};
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
        /* Format: "chat <max_tokens> <user message>" */
        const char* rest = cmd + 5;
        int n = 256;
        /* Try to parse token count */
        char* endp = NULL;
        long val = strtol(rest, &endp, 10);
        const char* msg = rest;
        if (endp && endp != rest && *endp == ' ') {
            n = (int)val;
            msg = endp + 1;
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
