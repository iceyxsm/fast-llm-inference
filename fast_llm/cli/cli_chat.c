/*
 * fllm CLI - Chat interface with model selection & parameter config
 *
 * Flow:
 *   1. Select model from downloaded models
 *   2. Configure parameters (temperature, top_p, top_k, context, layers, etc.)
 *   3. Start daemon with selected model
 *   4. Interactive chat loop
 */
#include "cli.h"

/* ── Default config ───────────────────────────────────────────────── */

chat_config_t chat_default_config(void) {
    chat_config_t c;
    memset(&c, 0, sizeof(c));
    c.max_tokens = 256;
    c.temperature = 0.7f;
    c.top_p = 0.9f;
    c.top_k = 40;
    c.context_window = 2048;
    c.max_layers = 0;  /* 0 = all */
    c.repeat_penalty_on = 1;
    c.repeat_penalty = 1.1f;
    return c;
}

/* ── Config persistence ───────────────────────────────────────────── */

/* Config files stored as: ~/.fllm/configs/<model_name>.cfg (binary) */

static void get_config_dir(char* buf, int sz) {
    char fdir[512];
    cli_get_fllm_dir(fdir, sizeof(fdir));
    snprintf(buf, sz, "%s%cconfigs", fdir, PATH_SEP);
}

static void get_config_path(const char* model_name, char* buf, int sz) {
    char cdir[512];
    get_config_dir(cdir, sizeof(cdir));
    snprintf(buf, sz, "%s%c%s.cfg", cdir, PATH_SEP, model_name);
}

void chat_save_config(const chat_config_t* cfg) {
    if (cfg->model_name[0] == '\0') return;
    char cdir[512], path[512];
    get_config_dir(cdir, sizeof(cdir));
    MKDIR_P(cdir);
    get_config_path(cfg->model_name, path, sizeof(path));
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(cfg, sizeof(*cfg), 1, f); fclose(f); }
}

int chat_load_config(const char* model_name, chat_config_t* cfg) {
    char path[512];
    get_config_path(model_name, path, sizeof(path));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int ok = (fread(cfg, sizeof(*cfg), 1, f) == 1);
    fclose(f);
    return ok;
}

/* ── Default model persistence ────────────────────────────────────── */

static void get_default_model_path(char* buf, int sz) {
    char fdir[512];
    cli_get_fllm_dir(fdir, sizeof(fdir));
    snprintf(buf, sz, "%s%cdefault_model.txt", fdir, PATH_SEP);
}

void chat_set_default_model(const char* model_path) {
    char path[512];
    get_default_model_path(path, sizeof(path));
    cli_ensure_dirs();
    FILE* f = fopen(path, "w");
    if (f) { fprintf(f, "%s\n", model_path); fclose(f); }
}

int chat_get_default_model(char* out, int sz) {
    char path[512];
    get_default_model_path(path, sizeof(path));
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(out, sz, f)) { fclose(f); return 0; }
    fclose(f);
    out[strcspn(out, "\n")] = '\0';
    return (strlen(out) > 0 && cli_file_exists(out));
}

/* ── Quick Start ──────────────────────────────────────────────────── */

void chat_quick_start(void) {
    char model_path[512] = {0};

    if (!chat_get_default_model(model_path, sizeof(model_path))) {
        /* No default set — try first available model */
        if (!find_local_model(model_path, sizeof(model_path))) {
            cli_cprint("  No default model set and no models downloaded.\n", CLR_YELLOW);
            printf("  Use 'Browse Models' to download one, or 'Chat' to set a default.\n\n");
            return;
        }
    }

    /* Load saved config for this model, or use defaults */
    chat_config_t cfg = chat_default_config();
    strncpy(cfg.model_path, model_path, sizeof(cfg.model_path)-1);

    const char* fn = strrchr(model_path, PATH_SEP);
    if (fn) fn++; else fn = model_path;
    strncpy(cfg.model_name, fn, sizeof(cfg.model_name)-1);
    char* ext = strstr(cfg.model_name, ".gguf");
    if (ext) *ext = '\0';

    /* Try loading saved config */
    chat_config_t saved;
    if (chat_load_config(cfg.model_name, &saved)) {
        /* Restore saved params but keep the model path fresh */
        char mp[512], mn[64];
        strncpy(mp, cfg.model_path, sizeof(mp)-1); mp[sizeof(mp)-1]='\0';
        strncpy(mn, cfg.model_name, sizeof(mn)-1); mn[sizeof(mn)-1]='\0';
        cfg = saved;
        strncpy(cfg.model_path, mp, sizeof(cfg.model_path)-1);
        strncpy(cfg.model_name, mn, sizeof(cfg.model_name)-1);
    }

    printf("\n  Quick Start: ");
    cli_cprint(cfg.model_name, CLR_GREEN);
    printf("\n");
    fflush(stdout);

    /* Go straight to chat — skip config screen */
    chat_run(&cfg);
}

/* ── Model selection ──────────────────────────────────────────────── */

static int select_model(chat_config_t* cfg) {
    char locals[16][512];
    int nlocal = list_local_models(locals, 16);

    if (nlocal == 0) {
        cli_cprint("  No models downloaded.\n", CLR_YELLOW);
        printf("  Use 'Browse Models' from the main menu first.\n\n");
        return 0;
    }

    cli_separator();
    cli_cprint("  SELECT MODEL\n", CLR_CYAN);
    cli_separator();
    fflush(stdout); cli_delay(100);
    printf("\n");

    for (int i = 0; i < nlocal; i++) {
        const char* fname = strrchr(locals[i], PATH_SEP);
        if (fname) fname++; else fname = locals[i];

        FILE* f = fopen(locals[i], "rb");
        double sz = 0;
        if (f) { fseek(f, 0, SEEK_END); sz = ftell(f) / (1024.0*1024.0); fclose(f); }

        char c_sz[16];
        if (sz >= 1024) snprintf(c_sz, sizeof(c_sz), "%.1f GB", sz/1024.0);
        else snprintf(c_sz, sizeof(c_sz), "%.0f MB", sz);

        /* Show clean name */
        char disp[32];
        strncpy(disp, fname, 31); disp[31] = '\0';
        char* ext = strstr(disp, ".gguf"); if (ext) *ext = '\0';
        /* Truncate to 28 chars */
        if (strlen(disp) > 28) { disp[25] = '.'; disp[26] = '.'; disp[27] = '.'; disp[28] = '\0'; }

        printf("  ");
        cli_color(CLR_GREEN);
        printf("[%d]", i+1);
        cli_reset();
        printf(" %-30s %s\n", disp, c_sz);
        fflush(stdout); cli_delay(50);
    }

    printf("\n  Select model (0 = back): ");
    int choice = cli_prompt_choice(nlocal);
    if (choice <= 0) return 0;

    strncpy(cfg->model_path, locals[choice-1], sizeof(cfg->model_path)-1);
    const char* fn = strrchr(cfg->model_path, PATH_SEP);
    if (fn) fn++; else fn = cfg->model_path;
    strncpy(cfg->model_name, fn, sizeof(cfg->model_name)-1);
    /* Trim .gguf */
    char* ext = strstr(cfg->model_name, ".gguf");
    if (ext) *ext = '\0';

    return 1;
}

/* ── Parameter configuration ──────────────────────────────────────── */

static void show_config(const chat_config_t* cfg) {
    cli_separator();
    cli_cprint("  CHAT SETTINGS\n", CLR_CYAN);
    cli_separator();
    fflush(stdout); cli_delay(100);
    printf("\n");

    printf("  Model: ");
    cli_cprint(cfg->model_name, CLR_GREEN);
    printf("\n\n");
    fflush(stdout); cli_delay(60);

    printf("  ");
    cli_cprint("[1]", CLR_GREEN);
    printf(" Max Tokens      %d\n", cfg->max_tokens);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[2]", CLR_GREEN);
    printf(" Temperature     %.2f\n", cfg->temperature);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[3]", CLR_GREEN);
    printf(" Top-P           %.2f\n", cfg->top_p);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[4]", CLR_GREEN);
    printf(" Top-K           %d\n", cfg->top_k);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[5]", CLR_GREEN);
    printf(" Context Window  %d\n", cfg->context_window);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[6]", CLR_GREEN);
    printf(" Max Layers      %s\n", cfg->max_layers > 0 ? "" : "All");
    if (cfg->max_layers > 0) printf("%d", cfg->max_layers);
    fflush(stdout); cli_delay(40);

    printf("  ");
    cli_cprint("[7]", CLR_GREEN);
    printf(" Repeat Penalty  %s", cfg->repeat_penalty_on ? "On" : "Off");
    if (cfg->repeat_penalty_on) printf(" (%.2f)", cfg->repeat_penalty);
    printf("\n");
    fflush(stdout); cli_delay(40);
}

static int read_int(const char* prompt, int current) {
    printf("  %s [%d]: ", prompt, current);
    fflush(stdout);
    char buf[32]={0};
    if (!fgets(buf, sizeof(buf), stdin)) return current;
    buf[strcspn(buf, "\n")] = '\0';
    if (strlen(buf) == 0) return current;
    return atoi(buf);
}

static float read_float(const char* prompt, float current) {
    printf("  %s [%.2f]: ", prompt, current);
    fflush(stdout);
    char buf[32]={0};
    if (!fgets(buf, sizeof(buf), stdin)) return current;
    buf[strcspn(buf, "\n")] = '\0';
    if (strlen(buf) == 0) return current;
    return (float)atof(buf);
}

void chat_configure(chat_config_t* cfg) {
    while (1) {
        cli_clear();
        show_config(cfg);

        printf("\n");
        printf("  Enter number to change, ");
        cli_cprint("[S]", CLR_GREEN);
        printf(" Start chat, ");
        cli_cprint("[D]", CLR_CYAN);
        printf(" Set as default, ");
        cli_cprint("[M]", CLR_RED);
        printf(" Back\n");
        printf("\n  > ");
        fflush(stdout);

        char buf[16]={0};
        if (!fgets(buf, sizeof(buf), stdin)) return;
        buf[strcspn(buf, "\n")] = '\0';

        if (buf[0]=='s'||buf[0]=='S') {
            chat_save_config(cfg);
            return;
        }
        if (buf[0]=='d'||buf[0]=='D') {
            chat_save_config(cfg);
            chat_set_default_model(cfg->model_path);
            cli_cprint("\n  Saved as default model!\n\n", CLR_GREEN);
            fflush(stdout); cli_delay(500);
            continue;
        }
        if (buf[0]=='m'||buf[0]=='M'||buf[0]=='0') { cfg->model_path[0]='\0'; return; }

        int c = atoi(buf);
        printf("\n");
        switch (c) {
            case 1:
                cfg->max_tokens = read_int("Max tokens (1-2048)", cfg->max_tokens);
                if (cfg->max_tokens < 1) cfg->max_tokens = 1;
                if (cfg->max_tokens > 2048) cfg->max_tokens = 2048;
                break;
            case 2:
                cfg->temperature = read_float("Temperature (0.0-2.0)", cfg->temperature);
                if (cfg->temperature < 0) cfg->temperature = 0;
                if (cfg->temperature > 2.0f) cfg->temperature = 2.0f;
                break;
            case 3:
                cfg->top_p = read_float("Top-P (0.0-1.0)", cfg->top_p);
                if (cfg->top_p < 0) cfg->top_p = 0;
                if (cfg->top_p > 1.0f) cfg->top_p = 1.0f;
                break;
            case 4:
                cfg->top_k = read_int("Top-K (1-200)", cfg->top_k);
                if (cfg->top_k < 1) cfg->top_k = 1;
                if (cfg->top_k > 200) cfg->top_k = 200;
                break;
            case 5:
                cfg->context_window = read_int("Context window (256-131072)", cfg->context_window);
                if (cfg->context_window < 256) cfg->context_window = 256;
                if (cfg->context_window > 131072) cfg->context_window = 131072;
                break;
            case 6: {
                int v = read_int("Max layers (0 = all)", cfg->max_layers);
                cfg->max_layers = (v < 0) ? 0 : v;
                break;
            }
            case 7:
                cfg->repeat_penalty_on = !cfg->repeat_penalty_on;
                if (cfg->repeat_penalty_on) {
                    cfg->repeat_penalty = read_float("Penalty value (1.0-2.0)", cfg->repeat_penalty);
                    if (cfg->repeat_penalty < 1.0f) cfg->repeat_penalty = 1.0f;
                    if (cfg->repeat_penalty > 2.0f) cfg->repeat_penalty = 2.0f;
                }
                break;
        }
    }
}

/* ── Chat loop ────────────────────────────────────────────────────── */

void chat_run(chat_config_t* cfg) {
    /* Step 1: Select model if not set */
    if (cfg->model_path[0] == '\0') {
        if (!select_model(cfg)) return;

        /* Try loading saved config for this model */
        chat_config_t saved;
        if (chat_load_config(cfg->model_name, &saved)) {
            char mp[512], mn[64];
            strncpy(mp, cfg->model_path, sizeof(mp)-1); mp[sizeof(mp)-1]='\0';
            strncpy(mn, cfg->model_name, sizeof(mn)-1); mn[sizeof(mn)-1]='\0';
            *cfg = saved;
            strncpy(cfg->model_path, mp, sizeof(cfg->model_path)-1);
            strncpy(cfg->model_name, mn, sizeof(cfg->model_name)-1);
        }
    }

    /* Step 2: Configure parameters */
    chat_configure(cfg);
    if (cfg->model_path[0] == '\0') return; /* user cancelled */

    /* Step 3: Start or connect to daemon */
    int port;
    int need_start = 1;

    if (daemon_is_running(&port)) {
        /* Check if same model is loaded */
        char resp[4096]={0};
        if (daemon_send(port, "status", resp, sizeof(resp))) {
            /* If a different model is loaded, stop and restart */
            if (strstr(resp, cfg->model_name)) {
                need_start = 0; /* same model already running */
            } else {
                printf("\n  Different model loaded. Restarting daemon...\n");
                daemon_stop();
                SLEEP_MS(2000);
            }
        }
    }

    if (need_start) {
        cli_clear();
        printf("\n  Loading model: ");
        cli_cprint(cfg->model_name, CLR_GREEN);
        printf("\n");

        daemon_start_detached(cfg->model_path);

        printf("  Starting daemon");
        fflush(stdout);
        int connected = 0;
        for (int i = 0; i < 60; i++) {
            SLEEP_MS(1000);
            printf(".");
            fflush(stdout);
            if (daemon_is_running(&port)) {
                char resp[4096];
                if (daemon_send(port, "status", resp, sizeof(resp))) {
                    connected = 1;
                    break;
                }
            }
        }

        if (!connected) {
            printf("\n  Daemon failed to start. Check ~/.fllm/fllm.log\n");
            printf("  Press Enter...");
            fflush(stdout);
            { char _b[4]; fgets(_b, sizeof(_b), stdin); }
            return;
        }
        printf(" ready!\n");
    }

    /* Step 4: Chat interface */
    cli_clear();
    cli_separator();
    cli_cprint("  CHAT\n", CLR_CYAN);
    cli_separator();
    printf("\n");
    printf("  Model: ");
    cli_cprint(cfg->model_name, CLR_GREEN);
    printf("\n");
    printf("  Tokens: %d  Temp: %.2f  Top-P: %.2f  Top-K: %d\n",
           cfg->max_tokens, cfg->temperature, cfg->top_p, cfg->top_k);
    printf("\n");
    printf("  Commands:\n");
    printf("    /config    Change parameters\n");
    printf("    /status    Model info\n");
    printf("    /clear     Clear screen\n");
    printf("    /quit      Exit chat (daemon keeps running)\n");
    printf("    /stop      Exit chat and stop daemon\n");
    printf("\n");
    cli_separator();
    printf("\n");
    fflush(stdout);

    while (1) {
        cli_color(CLR_GREEN);
        printf("  You> ");
        cli_reset();
        fflush(stdout);

        char input[1024]={0};
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = '\0';
        if (strlen(input) == 0) continue;

        /* Handle commands */
        if (input[0] == '/') {
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
                printf("\n  Disconnected. Daemon still running.\n");
                printf("  Use 'fllm -off' or menu option 5 to stop it.\n\n");
                break;
            }
            if (strcmp(input, "/stop") == 0) {
                printf("\n  Stopping daemon...\n");
                daemon_stop();
                printf("  Done.\n\n");
                break;
            }
            if (strcmp(input, "/config") == 0) {
                chat_configure(cfg);
                if (cfg->model_path[0] == '\0') break;
                chat_save_config(cfg);
                cli_clear();
                cli_separator();
                cli_cprint("  CHAT\n", CLR_CYAN);
                cli_separator();
                printf("\n  Settings updated: Tokens=%d Temp=%.2f Top-P=%.2f Top-K=%d\n\n",
                       cfg->max_tokens, cfg->temperature, cfg->top_p, cfg->top_k);
                continue;
            }
            if (strcmp(input, "/status") == 0) {
                char resp[4096]={0};
                if (daemon_send(port, "status", resp, sizeof(resp)))
                    printf("\n  %s\n", resp);
                else
                    printf("\n  Lost connection to daemon.\n\n");
                continue;
            }
            if (strcmp(input, "/clear") == 0) {
                cli_clear();
                cli_separator();
                cli_cprint("  CHAT\n", CLR_CYAN);
                cli_separator();
                printf("\n");
                continue;
            }
            printf("  Unknown command. Type /quit, /config, /status, /clear, /stop\n\n");
            continue;
        }

        /* Send generate command to daemon */
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "gen %d", cfg->max_tokens);

        char resp[4096]={0};
        printf("\n  ");
        cli_color(CLR_CYAN);
        printf("AI> ");
        cli_reset();
        fflush(stdout);

        if (daemon_send(port, cmd, resp, sizeof(resp))) {
            printf("%s\n\n", resp);
        } else {
            cli_cprint("Connection lost.\n\n", CLR_RED);
            break;
        }
    }
}
