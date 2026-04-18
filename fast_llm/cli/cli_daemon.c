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
    char path[512];
    cli_get_pid_path(path, sizeof(path));
    FILE* f = fopen(path, "w");
    if (f) { fprintf(f, "%d\n%d\n", GETPID(), DAEMON_PORT); fclose(f); }
}

void daemon_remove_pid(void) {
    char path[512];
    cli_get_pid_path(path, sizeof(path));
    remove(path);
}

int daemon_read_pid(int* pid, int* port) {
    char path[512];
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

/* ── Generate tokens (used by daemon) ─────────────────────────────── */

static void run_generate(transformer_model_t* model, int ntok, char* out, int osz) {
    int vocab = model->config.vocab_size;
    float* logits = (float*)aligned_malloc(vocab * sizeof(float), 64);
    int* ctx = (int*)malloc(2048 * sizeof(int));
    int ctx_len = 0;

    int seed[] = { 1, 450, 6593, 310, 2834, 338 };
    for (int i = 0; i < 6; i++) ctx[ctx_len++] = seed[i];

    double start = cli_time_sec();
    int gen = 0;

    for (int t = 0; t < ntok; t++) {
        int next = 0;
        model_forward(model, ctx, ctx_len, logits, &next);
        float best = logits[0]; int bid = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > best) { best = logits[v]; bid = v; }
        next = bid;
        if (ctx_len < 2048) ctx[ctx_len++] = next;
        else { memmove(ctx, ctx+64, (ctx_len-64)*sizeof(int)); ctx_len -= 64; ctx[ctx_len++] = next; }
        gen++;
        if (next == 2) break;
    }

    double elapsed = cli_time_sec() - start;
    snprintf(out, osz,
        "Generated %d tokens in %.2f sec\n"
        "Speed: %.2f tok/sec (%.2f ms/token)\n"
        "Model: %s (%d layers, Q%d)\n",
        gen, elapsed, gen/elapsed, (elapsed/gen)*1000.0,
        model->config.model_name, model->config.num_layers, model->config.quant_bits);

    free(ctx); aligned_free(logits);
}

/* ── Handle daemon command ────────────────────────────────────────── */

static void handle_cmd(const char* raw, char* resp, int rsz) {
    char cmd[256];
    strncpy(cmd, raw, sizeof(cmd)-1); cmd[sizeof(cmd)-1] = '\0';
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
        if (n < 1) n = 50; if (n > 500) n = 500;
        run_generate(g_model, n, resp, rsz);
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

        char buf[1024] = {0};
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
    char exe[512], cmdline[1024];
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
    char logpath[512];
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
