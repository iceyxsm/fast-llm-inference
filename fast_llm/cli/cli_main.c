/*
 * fllm CLI - Main entry point & interactive menu
 *
 * Usage:
 *   fllm              Interactive menu
 *   fllm -off         Stop daemon
 *   fllm --status     Check daemon
 *   fllm --_daemon    (internal) run as daemon child
 */
#include "cli.h"
#include "cpu_features.h"

/* ── Catalog state ────────────────────────────────────────────────── */
static model_entry_t g_catalog[MAX_CATALOG];
static int g_catalog_count = 0;

/* ── Main menu ────────────────────────────────────────────────────── */

static void show_menu(const hw_specs_t* specs) {
    cli_separator();
    cli_cprint("  MAIN MENU\n", CLR_CYAN);
    cli_separator();
    fflush(stdout); cli_delay(100);
    printf("\n");

    printf("  ");
    cli_cprint("[1]", CLR_GREEN);
    printf(" Check Specs       Detect your hardware");
    if (specs->valid) { printf("  ("); cli_stars(specs->stars); printf(")"); }
    printf("\n"); fflush(stdout); cli_delay(60);

    printf("  ");
    cli_cprint("[2]", CLR_GREEN);
    printf(" Browse Models     Fetch & browse available models\n");
    fflush(stdout); cli_delay(60);

    printf("  ");
    cli_cprint("[3]", CLR_GREEN);
    printf(" My Models         View downloaded models\n");
    fflush(stdout); cli_delay(60);

    printf("  ");
    cli_cprint("[4]", CLR_GREEN);
    printf(" Run Model         Load a model & start inference\n");
    fflush(stdout); cli_delay(60);

    printf("  ");
    cli_cprint("[5]", CLR_GREEN);
    printf(" Daemon Status     Check / stop background server\n");
    fflush(stdout); cli_delay(60);

    printf("\n");
    printf("  ");
    cli_cprint("[0]", CLR_RED);
    printf(" Exit\n");
    fflush(stdout);
}

/* ── Run model flow ───────────────────────────────────────────────── */

static void run_model_flow(const char* explicit_model) {
    int port;

    /* Already running? */
    if (daemon_is_running(&port)) {
        printf("\n  Daemon already running.\n");
        daemon_interactive(port);
        return;
    }

    char model_path[512] = {0};

    if (explicit_model && explicit_model[0]) {
        strncpy(model_path, explicit_model, sizeof(model_path) - 1);
    } else {
        /* List local models and let user pick */
        char locals[16][512];
        int nlocal = list_local_models(locals, 16);

        if (nlocal == 0) {
            printf("\n");
            cli_cprint("  No models downloaded yet.\n", CLR_YELLOW);
            printf("  Use 'Browse Models' to fetch and download one.\n\n");
            return;
        }

        if (nlocal == 1) {
            strncpy(model_path, locals[0], sizeof(model_path) - 1);
        } else {
            printf("\n  Local models:\n\n");
            for (int i = 0; i < nlocal; i++) {
                /* Show just the filename */
                const char* fname = strrchr(locals[i], PATH_SEP);
                if (fname) fname++; else fname = locals[i];
                printf("  ");
                cli_cprint("[", CLR_GREEN);
                printf("%d", i + 1);
                cli_cprint("]", CLR_GREEN);
                printf(" %s\n", fname);
            }
            printf("\n  Select model (0 = back): ");
            int choice = cli_prompt_choice(nlocal);
            if (choice <= 0) return;
            strncpy(model_path, locals[choice - 1], sizeof(model_path) - 1);
        }
    }

    if (!cli_file_exists(model_path)) {
        cli_cprint("  Error: file not found.\n\n", CLR_RED);
        return;
    }

    /* Show what we're loading */
    const char* fname = strrchr(model_path, PATH_SEP);
    if (fname) fname++; else fname = model_path;
    printf("\n  Loading: %s\n", fname);
    printf("  Starting daemon...\n");

    daemon_start_detached(model_path);

    /* Wait for daemon */
    printf("  Waiting");
    fflush(stdout);
    for (int i = 0; i < 60; i++) {
        SLEEP_MS(1000);
        printf(".");
        fflush(stdout);
        if (daemon_is_running(&port)) {
            char resp[4096];
            if (daemon_send(port, "status", resp, sizeof(resp))) {
                printf(" ready!\n");
                daemon_interactive(port);
                return;
            }
        }
    }
    printf("\n  Daemon didn't respond. Check ~/.fllm/fllm.log\n\n");
}

/* ── Daemon status flow ───────────────────────────────────────────── */

static void status_flow(void) {
    int port;
    printf("\n");
    if (!daemon_is_running(&port)) {
        cli_cprint("  Daemon: not running\n\n", CLR_YELLOW);
        return;
    }
    char resp[4096];
    if (daemon_send(port, "status", resp, sizeof(resp))) {
        cli_cprint("  Daemon: running\n", CLR_GREEN);
        printf("  %s\n", resp);
    } else {
        cli_cprint("  Daemon: PID exists but not responding\n\n", CLR_RED);
    }

    if (cli_prompt_yn("Stop daemon?")) {
        daemon_stop();
        printf("\n");
    }
}

/* ── My Models: show all downloaded models ─────────────────────────── */

static void my_models_flow(void) {
    char locals[16][512];
    int nlocal = list_local_models(locals, 16);

    cli_separator();
    cli_cprint("  MY MODELS\n", CLR_CYAN);
    cli_separator();
    fflush(stdout); cli_delay(100);
    printf("\n");

    if (nlocal == 0) {
        cli_cprint("  No models downloaded yet.\n", CLR_YELLOW);
        printf("  Use 'Browse Models' to find and download models.\n\n");
        return;
    }

    printf("  %-5s %-30s %s\n", "#", "Filename", "Size");
    printf("  %-5s %-30s %s\n", "---", "------------------------------", "--------");
    fflush(stdout); cli_delay(100);

    for (int i = 0; i < nlocal; i++) {
        const char* fname = strrchr(locals[i], PATH_SEP);
        if (fname) fname++; else fname = locals[i];

        /* Get file size */
        FILE* f = fopen(locals[i], "rb");
        double size_mb = 0;
        if (f) {
            fseek(f, 0, SEEK_END);
            size_mb = ftell(f) / (1024.0 * 1024.0);
            fclose(f);
        }

        char c_size[16];
        if (size_mb >= 1024) snprintf(c_size, sizeof(c_size), "%.1f GB", size_mb / 1024.0);
        else snprintf(c_size, sizeof(c_size), "%.0f MB", size_mb);

        printf("  ");
        cli_color(CLR_GREEN);
        printf("[%2d]", i + 1);
        cli_reset();

        /* Truncate filename to 30 chars */
        char disp[31];
        strncpy(disp, fname, 30); disp[30] = '\0';
        printf(" %-30s %s\n", disp, c_size);
        fflush(stdout); cli_delay(70);
    }
    printf("\n");
}

/* ── Custom URL download ──────────────────────────────────────────── */

static void custom_url_flow(void) {
    printf("\n");
    cli_cprint("  CUSTOM MODEL DOWNLOAD\n", CLR_CYAN);
    printf("\n  Paste a direct URL to a .gguf file:\n");
    printf("  > ");
    fflush(stdout);

    char url[1024] = {0};
    if (!fgets(url, sizeof(url), stdin)) return;
    url[strcspn(url, "\n")] = '\0';
    if (strlen(url) < 10) { printf("  Invalid URL.\n"); return; }

    /* Extract filename from URL */
    const char* last_slash = strrchr(url, '/');
    const char* fname = last_slash ? last_slash + 1 : "custom_model.gguf";

    /* Check it ends with .gguf */
    if (!strstr(fname, ".gguf")) {
        cli_cprint("  Warning: ", CLR_YELLOW);
        printf("URL doesn't end with .gguf. Continue? ");
        if (!cli_prompt_yn("Download anyway?")) return;
    }

    char mdir[512], dest[512];
    cli_get_models_dir(mdir, sizeof(mdir));
    cli_ensure_dirs();
    snprintf(dest, sizeof(dest), "%s%c%s", mdir, PATH_SEP, fname);

    if (cli_file_exists(dest)) {
        printf("\n  ");
        cli_cprint(fname, CLR_GREEN);
        printf(" already exists.\n\n");
        return;
    }

    printf("\n  Downloading: %s\n", fname);
    printf("  To: %s\n\n", dest);

    if (download_file(url, dest)) {
        cli_cprint("  Download complete!\n\n", CLR_GREEN);
    } else {
        cli_cprint("  Download failed.\n\n", CLR_RED);
    }
}

/* ── Browse models flow ───────────────────────────────────────────── */

static void browse_flow(hw_specs_t* specs) {
    while (1) {
        cli_clear();

        /* Step 1: Select model type */
        cli_separator();
        cli_cprint("  SELECT MODEL TYPE\n", CLR_CYAN);
        cli_separator();
        fflush(stdout); cli_delay(100);
        printf("\n");

        printf("  "); cli_cprint("[1]", CLR_GREEN); printf(" Chat / Text         General purpose LLMs\n");
        fflush(stdout); cli_delay(60);
        printf("  "); cli_cprint("[2]", CLR_GREEN); printf(" Vision              Image understanding models\n");
        fflush(stdout); cli_delay(60);
        printf("  "); cli_cprint("[3]", CLR_GREEN); printf(" Code                Code-specialized models\n");
        fflush(stdout); cli_delay(60);
        printf("  "); cli_cprint("[4]", CLR_GREEN); printf(" Voice (TTS)         Text-to-speech models\n");
        fflush(stdout); cli_delay(60);
        printf("  "); cli_cprint("[5]", CLR_GREEN); printf(" Speech-to-Text      Whisper and similar\n");
        fflush(stdout); cli_delay(60);
        printf("\n");
        printf("  "); cli_cprint("[R]", CLR_CYAN);  printf(" Refresh catalog from internet\n");
        fflush(stdout); cli_delay(40);
        printf("  "); cli_cprint("[M]", CLR_RED);   printf(" Main menu\n");
        fflush(stdout);

        printf("\n  > ");
        fflush(stdout);
        char buf[16]={0};
        if (!fgets(buf, sizeof(buf), stdin)) return;
        buf[strcspn(buf, "\n")] = '\0';

        if (buf[0]=='0'||buf[0]=='b'||buf[0]=='B'||buf[0]=='m'||buf[0]=='M') return;

        if (buf[0]=='r'||buf[0]=='R') {
            cli_clear();
            g_catalog_count = catalog_refresh(g_catalog, MAX_CATALOG, MTYPE_LLM);
            continue;
        }

        model_type_t type;
        int tc = atoi(buf);
        switch (tc) {
            case 1: type = MTYPE_LLM; break;
            case 2: type = MTYPE_VISION; break;
            case 3: type = MTYPE_CODE; break;
            case 4: type = MTYPE_VOICE; break;
            case 5: type = MTYPE_STT; break;
            default: continue;
        }

        /* Load or fetch catalog */
        if (g_catalog_count == 0)
            g_catalog_count = catalog_load_cache(g_catalog, MAX_CATALOG);

        int have_type = 0;
        for (int i = 0; i < g_catalog_count; i++)
            if (g_catalog[i].type == type) { have_type = 1; break; }

        if (!have_type) {
            cli_clear();
            model_entry_t tmp[MAX_CATALOG];
            int nc = catalog_fetch(tmp, MAX_CATALOG, type);
            for (int i = 0; i < nc && g_catalog_count < MAX_CATALOG; i++)
                g_catalog[g_catalog_count++] = tmp[i];
            if (g_catalog_count > 0) catalog_save_cache(g_catalog, g_catalog_count);
        }

        /* Clear and show size filter + table on same screen */
        cli_clear();
        catalog_browse(g_catalog, g_catalog_count, specs, type);

        /* Action options after table */
        printf("  Enter numbers to download (e.g. 1,3,5)\n");
        printf("  ");
        cli_cprint("[L]", CLR_CYAN);
        printf(" Load more  ");
        cli_cprint("[U]", CLR_CYAN);
        printf(" Custom URL  ");
        cli_cprint("[M]", CLR_RED);
        printf(" Main menu\n");
        printf("\n  > ");
        fflush(stdout);
        char dbuf[256]={0};
        if (!fgets(dbuf, sizeof(dbuf), stdin)) return;
        dbuf[strcspn(dbuf, "\n")] = '\0';
        if (dbuf[0]=='m'||dbuf[0]=='M'||dbuf[0]=='0') continue; /* back to type selector → main menu */

        /* Load more — re-fetch from API */
        if (dbuf[0]=='l'||dbuf[0]=='L') {
            cli_clear();
            model_entry_t tmp[MAX_CATALOG];
            int nc = catalog_fetch(tmp, MAX_CATALOG, type);
            for (int i = 0; i < nc && g_catalog_count < MAX_CATALOG; i++) {
                /* Only add if not already present */
                int dup = 0;
                for (int j = 0; j < g_catalog_count; j++)
                    if (strcmp(g_catalog[j].full_name, tmp[i].full_name)==0) { dup=1; break; }
                if (!dup) g_catalog[g_catalog_count++] = tmp[i];
            }
            if (g_catalog_count > 0) catalog_save_cache(g_catalog, g_catalog_count);
            continue;
        }

        /* Custom URL */
        if (dbuf[0]=='u'||dbuf[0]=='U') {
            custom_url_flow();
            printf("  Press Enter to continue...");
            fflush(stdout);
            { char _b[4]; fgets(_b, sizeof(_b), stdin); }
            continue;
        }

        /* Build filtered index */
        int idx[MAX_CATALOG]; int fcount=0;
        for (int i=0;i<g_catalog_count;i++)
            if (g_catalog[i].type == type) idx[fcount++]=i;

        /* Parse comma-separated */
        char* tok = strtok(dbuf, ", ");
        while (tok) {
            int choice = atoi(tok);
            if (choice >= 1 && choice <= fcount) {
                model_entry_t* m = &g_catalog[idx[choice-1]];
                char mdir[512], dest[512];
                cli_get_models_dir(mdir, sizeof(mdir));
                snprintf(dest, sizeof(dest), "%s%c%s", mdir, PATH_SEP, m->filename);

                if (cli_file_exists(dest)) {
                    printf("\n  "); cli_cprint(m->short_name, CLR_GREEN);
                    printf(" already downloaded.\n");
                } else {
                    if (specs && specs->valid && specs->tier < m->min_tier) {
                        printf("\n  "); cli_cprint("Note: ", CLR_YELLOW);
                        printf("%s needs %s tier.\n", m->short_name, cli_tier_name(m->min_tier));
                    }
                    download_model(m);
                }
            }
            tok = strtok(NULL, ", ");
        }

        printf("\n  Press Enter to continue...");
        fflush(stdout);
        { char _b[4]; fgets(_b, sizeof(_b), stdin); }
    }
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));

    char model_path[512] = {0};
    int do_off = 0, do_status = 0, is_daemon = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-off") == 0 || strcmp(argv[i], "--off") == 0 || strcmp(argv[i], "--stop") == 0)
            do_off = 1;
        else if (strcmp(argv[i], "--status") == 0)
            do_status = 1;
        else if (strcmp(argv[i], "--model") == 0 && i+1 < argc)
            strncpy(model_path, argv[++i], sizeof(model_path)-1);
        else if (strcmp(argv[i], "--_daemon") == 0)
            is_daemon = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            cli_banner();
            printf("  Usage:\n");
            printf("    fllm                 Interactive menu\n");
            printf("    fllm -off            Stop background daemon\n");
            printf("    fllm --status        Check daemon status\n");
            printf("    fllm --model <path>  Start with specific model\n");
            printf("    fllm -h              This help\n\n");
            return 0;
        }
    }

    /* Internal: daemon child process */
    if (is_daemon) {
        daemon_serve(model_path);
        return 0;
    }

    cli_clear();

    /* Quick commands */
    if (do_off) { daemon_stop(); return 0; }

    if (do_status) { status_flow(); return 0; }

    /* If --model passed, go straight to run */
    if (model_path[0]) {
        run_model_flow(model_path);
        return 0;
    }

    /* Load saved specs */
    hw_specs_t specs;
    memset(&specs, 0, sizeof(specs));
    specs_load(&specs);

    /* Interactive menu loop */
    while (1) {
        cli_clear();
        show_menu(&specs);
        int choice = cli_prompt_choice(5);

        switch (choice) {
            case 1:
                cli_clear();
                specs_run_interactive();
                specs_load(&specs);
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                break;
            case 2:
                cli_clear();
                browse_flow(&specs);
                break;
            case 3:
                cli_clear();
                my_models_flow();
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                break;
            case 4:
                cli_clear();
                run_model_flow(NULL);
                break;
            case 5:
                cli_clear();
                status_flow();
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                break;
            case 0:
                cli_clear();
                printf("\n  Bye!\n\n");
                return 0;
            default:
                break;
        }
    }
}
