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

static void show_menu(const hw_specs_t* specs, int animate) {
    cli_separator();
    cli_cprint("  MAIN MENU\n", CLR_CYAN);
    cli_separator();
    if (animate) { fflush(stdout); cli_delay(100); }
    printf("\n");

    printf("  ");
    cli_cprint("[1]", CLR_GREEN);
    printf(" Quick Start       Chat with default model\n");
    if (animate) { fflush(stdout); cli_delay(60); }

    printf("  ");
    cli_cprint("[2]", CLR_GREEN);
    printf(" Check Specs       Detect your hardware");
    if (specs->valid) { printf("  ("); cli_stars(specs->stars); printf(")"); }
    printf("\n");
    if (animate) { fflush(stdout); cli_delay(60); }

    printf("  ");
    cli_cprint("[3]", CLR_GREEN);
    printf(" Browse Models     Fetch & browse available models\n");
    if (animate) { fflush(stdout); cli_delay(60); }

    printf("  ");
    cli_cprint("[4]", CLR_GREEN);
    printf(" My Models         View downloaded models\n");
    if (animate) { fflush(stdout); cli_delay(60); }

    printf("  ");
    cli_cprint("[5]", CLR_GREEN);
    printf(" Chat               Start chatting with a model\n");
    if (animate) { fflush(stdout); cli_delay(60); }

    printf("  ");
    cli_cprint("[6]", CLR_GREEN);
    printf(" Daemon Status     Check / stop background server\n");
    if (animate) { fflush(stdout); cli_delay(60); }

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

/* ── Helper: extract info from a gguf filename ────────────────────── */

static void parse_model_filename(const char* fname, char* name, int nsz,
                                  char* params, int psz, char* quant, int qsz) {
    char tmp[256];
    strncpy(tmp, fname, sizeof(tmp)-1); tmp[sizeof(tmp)-1]='\0';

    /* Remove .gguf extension */
    char* ext = strstr(tmp, ".gguf");
    if (ext) *ext = '\0';

    /* Extract quant */
    quant[0] = '\0';
    const char* qpats[] = {"Q4_K_M","Q4_K_S","Q4_0","Q4_1","Q5_K_M","Q5_K_S","Q3_K_M","Q3_K_L","Q6_K","Q8_0","f16","f32",NULL};
    for (int i=0; qpats[i]; i++) {
        char* q = strstr(tmp, qpats[i]);
        if (q) {
            strncpy(quant, qpats[i], qsz-1);
            /* Remove quant from name */
            if (q > tmp && (*(q-1)=='-'||*(q-1)=='_'||*(q-1)=='.')) q--;
            *q = '\0';
            break;
        }
    }

    /* Extract params (e.g. "7B", "1.1B") */
    params[0] = '\0';
    char* p = tmp;
    while (*p) {
        if ((*p>='0'&&*p<='9')||*p=='.') {
            char* start = p;
            while (*p && ((*p>='0'&&*p<='9')||*p=='.')) p++;
            if (*p=='B'||*p=='b') {
                int len = (int)(p - start + 1);
                if (len < psz) { memcpy(params, start, len); params[len]='\0'; }
                /* Remove from name */
                char* rm = start;
                if (rm > tmp && (*(rm-1)=='-'||*(rm-1)=='_')) rm--;
                memmove(rm, p+1, strlen(p+1)+1);
                p = rm;
                break;
            }
        }
        p++;
    }

    /* Clean name: replace - and _ with space, strip suffixes */
    for (p=tmp; *p; p++) if (*p=='-'||*p=='_') *p=' ';
    /* Remove common suffixes */
    const char* suf[] = {"Instruct","instruct","Chat","chat","GGUF","gguf","it",NULL};
    for (int i=0; suf[i]; i++) {
        char* s = strstr(tmp, suf[i]);
        if (s) {
            if (s > tmp && *(s-1)==' ') s--;
            *s = '\0';
        }
    }
    /* Trim */
    p = tmp; while (*p==' ') p++;
    int len = (int)strlen(p);
    while (len>0 && p[len-1]==' ') p[--len]='\0';

    strncpy(name, p, nsz-1); name[nsz-1]='\0';
}

/* ── My Models: show downloaded models with info ──────────────────── */

static void my_models_flow(void) {
    char locals[16][512];
    int nlocal = list_local_models(locals, 16);

    while (1) {
        cli_clear();
        cli_separator();
        cli_cprint("  MY MODELS\n", CLR_CYAN);
        cli_separator();
        fflush(stdout); cli_delay(100);
        printf("\n");

        if (nlocal == 0) {
            cli_cprint("  No models downloaded yet.\n", CLR_YELLOW);
            printf("  Use 'Browse Models' to find and download models.\n\n");
            printf("  "); cli_cprint("[M]", CLR_RED); printf(" Main menu\n");
            printf("\n  > "); fflush(stdout);
            char b[16]; if (!fgets(b,sizeof(b),stdin)) return;
            return;
        }

        /* Table header */
        printf("  %-5s %-20s %-7s %-7s %-7s %s\n",
               "#", "Name", "Params", "Size", "Quant", "Filename");
        printf("  %-5s %-20s %-7s %-7s %-7s %s\n",
               "---", "--------------------", "------", "-------", "------", "----------------------------");
        fflush(stdout); cli_delay(100);

        for (int i = 0; i < nlocal; i++) {
            const char* fpath = locals[i];
            const char* fname = strrchr(fpath, PATH_SEP);
            if (fname) fname++; else fname = fpath;

            /* Get file size */
            FILE* f = fopen(fpath, "rb");
            double size_mb = 0;
            if (f) { fseek(f, 0, SEEK_END); size_mb = ftell(f) / (1024.0*1024.0); fclose(f); }

            /* Parse info from filename */
            char mname[21], mparams[8], mquant[8];
            parse_model_filename(fname, mname, sizeof(mname), mparams, sizeof(mparams), mquant, sizeof(mquant));

            char c_size[8];
            if (size_mb >= 1024) snprintf(c_size, sizeof(c_size), "%.1fGB", size_mb/1024.0);
            else snprintf(c_size, sizeof(c_size), "%.0fMB", size_mb);

            /* Truncate filename for display */
            char fn_disp[29];
            strncpy(fn_disp, fname, 28); fn_disp[28]='\0';

            cli_color(CLR_GREEN);
            printf("  [%2d] ", i+1);
            cli_reset();
            printf("%-20s %-7s %-7s %-7s %s\n",
                   mname, mparams[0] ? mparams : "-", c_size,
                   mquant[0] ? mquant : "-", fn_disp);
            fflush(stdout); cli_delay(50);
        }

        printf("\n");
        printf("  Enter number for details, ");
        cli_cprint("[M]", CLR_RED);
        printf(" Main menu\n");
        printf("\n  > ");
        fflush(stdout);

        char buf[16]={0};
        if (!fgets(buf, sizeof(buf), stdin)) return;
        buf[strcspn(buf, "\n")] = '\0';

        if (buf[0]=='m'||buf[0]=='M'||buf[0]=='0'||buf[0]=='\0') return;

        int choice = atoi(buf);
        if (choice < 1 || choice > nlocal) continue;

        /* Show model details */
        const char* fpath = locals[choice-1];
        const char* fname = strrchr(fpath, PATH_SEP);
        if (fname) fname++; else fname = fpath;

        FILE* f = fopen(fpath, "rb");
        double size_mb = 0;
        if (f) { fseek(f, 0, SEEK_END); size_mb = ftell(f) / (1024.0*1024.0); fclose(f); }

        char mname[64], mparams[8], mquant[8];
        parse_model_filename(fname, mname, sizeof(mname), mparams, sizeof(mparams), mquant, sizeof(mquant));

        cli_clear();
        cli_separator();
        cli_cprint("  MODEL DETAILS\n", CLR_CYAN);
        cli_separator();
        fflush(stdout); cli_delay(150);

        printf("\n");
        printf("  Name:      "); cli_cprint(mname, CLR_GREEN); printf("\n");
        fflush(stdout); cli_delay(80);
        printf("  Filename:  %s\n", fname);
        fflush(stdout); cli_delay(80);
        if (mparams[0]) { printf("  Params:    %s\n", mparams); fflush(stdout); cli_delay(80); }
        if (mquant[0])  { printf("  Quant:     %s\n", mquant); fflush(stdout); cli_delay(80); }

        char c_size[16];
        if (size_mb >= 1024) snprintf(c_size, sizeof(c_size), "%.2f GB", size_mb/1024.0);
        else snprintf(c_size, sizeof(c_size), "%.1f MB", size_mb);
        printf("  Size:      %s\n", c_size);
        fflush(stdout); cli_delay(80);

        printf("  Path:      %s\n", fpath);
        fflush(stdout); cli_delay(80);

        /* Check if we have catalog info for this model */
        int found_in_catalog = 0;
        for (int i = 0; i < g_catalog_count; i++) {
            if (strcmp(g_catalog[i].filename, fname) == 0) {
                const model_entry_t* m = &g_catalog[i];
                found_in_catalog = 1;
                printf("\n");
                cli_cprint("  Catalog Info\n", CLR_CYAN);
                fflush(stdout); cli_delay(80);
                printf("  Family:    %s\n", m->family);
                fflush(stdout); cli_delay(60);
                printf("  Type:      %s\n", catalog_type_name(m->type));
                fflush(stdout); cli_delay(60);
                if (m->context_len > 0) {
                    if (m->context_len >= 131072) printf("  Context:   128K\n");
                    else if (m->context_len >= 32768) printf("  Context:   32K\n");
                    else if (m->context_len >= 8192) printf("  Context:   8K\n");
                    else printf("  Context:   %dK\n", m->context_len/1024);
                    fflush(stdout); cli_delay(60);
                }
                printf("  Tools:     %s\n", m->has_tools ? "Yes" : "No");
                fflush(stdout); cli_delay(60);
                printf("  Vision:    %s\n", m->has_vision ? "Yes" : "No");
                fflush(stdout); cli_delay(60);
                printf("  Code:      %s\n", m->has_code ? "Yes" : "No");
                fflush(stdout); cli_delay(60);
                if (m->license[0]) { printf("  License:   %s\n", m->license); fflush(stdout); cli_delay(60); }
                if (m->downloads > 0) { printf("  Downloads: %d\n", m->downloads); fflush(stdout); cli_delay(60); }
                printf("  Source:    %s\n", m->id);
                fflush(stdout); cli_delay(60);
                break;
            }
        }

        if (!found_in_catalog) {
            printf("\n  ");
            cli_cprint("(No catalog info — downloaded via custom URL or not in cache)\n", CLR_YELLOW);
        }

        printf("\n");
        printf("  Press Enter to go back...");
        fflush(stdout);
        { char _b[4]; fgets(_b, sizeof(_b), stdin); }
    }
}

/* ── Custom URL download ──────────────────────────────────────────── */

static void custom_url_flow(void) {
    printf("\n");
    cli_separator();
    cli_cprint("  CUSTOM MODEL DOWNLOAD\n", CLR_CYAN);
    cli_separator();
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
        printf("URL doesn't appear to be a .gguf file.\n");
        if (!cli_prompt_yn("Continue anyway?")) return;
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

    /* Show model info extracted from filename */
    printf("\n");
    cli_separator();
    cli_cprint("  MODEL INFO\n", CLR_CYAN);
    cli_separator();
    fflush(stdout); cli_delay(100);

    printf("\n");
    printf("  File:    %s\n", fname); fflush(stdout); cli_delay(60);
    printf("  Source:  %s\n", url); fflush(stdout); cli_delay(60);

    /* Try to extract info from filename */
    char tmp_name[128];
    strncpy(tmp_name, fname, sizeof(tmp_name)-1); tmp_name[sizeof(tmp_name)-1]='\0';

    /* Guess params from filename */
    double params = 0;
    const char* p = tmp_name;
    while (*p) {
        if ((*p>='0'&&*p<='9')||*p=='.') {
            double v = atof(p);
            while (*p&&((*p>='0'&&*p<='9')||*p=='.')) p++;
            if (*p=='B'||*p=='b') { params = v; break; }
        }
        p++;
    }
    if (params > 0) {
        printf("  Params:  %.1fB\n", params); fflush(stdout); cli_delay(60);
    }

    /* Guess quant */
    char quant[16] = {0};
    const char* qpats[] = {"Q4_K_M","Q4_K_S","Q4_0","Q5_K_M","Q3_K_M","Q6_K","Q8_0","f16",NULL};
    for (int i=0;qpats[i];i++) {
        if (strstr(fname, qpats[i])) { strncpy(quant, qpats[i], sizeof(quant)-1); break; }
    }
    if (quant[0]) {
        printf("  Quant:   %s\n", quant); fflush(stdout); cli_delay(60);
    }

    printf("  Save to: %s\n", dest); fflush(stdout); cli_delay(60);

    printf("\n");
    printf("  ");
    cli_cprint("[D]", CLR_GREEN);
    printf(" Download  ");
    cli_cprint("[M]", CLR_RED);
    printf(" Cancel\n");
    printf("\n  > ");
    fflush(stdout);

    char confirm[16] = {0};
    if (!fgets(confirm, sizeof(confirm), stdin)) return;
    if (confirm[0] != 'd' && confirm[0] != 'D') {
        printf("  Cancelled.\n\n");
        return;
    }

    printf("\n");
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

        /* Load cache if empty */
        if (g_catalog_count == 0)
            g_catalog_count = catalog_load_cache(g_catalog, MAX_CATALOG);

        /* Check if we have this type — if not, fetch inline */
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

            /* Re-check — if still nothing, skip */
            have_type = 0;
            for (int i = 0; i < g_catalog_count; i++)
                if (g_catalog[i].type == type) { have_type = 1; break; }
            if (!have_type) {
                printf("  No models found for this type.\n");
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                continue;
            }
        }

        /* Show catalog — size filter + paginated table */
        int cur_page = 0;
        while (1) {
            cli_clear();
            int action = catalog_browse(g_catalog, g_catalog_count, specs, type, cur_page);

            if (action == 'L') {
                /* Load more from API — stay in same type, jump to last page */
                cli_clear();
                int old_count = g_catalog_count;
                model_entry_t tmp[MAX_CATALOG];
                int nc = catalog_fetch(tmp, MAX_CATALOG, type);
                for (int i = 0; i < nc && g_catalog_count < MAX_CATALOG; i++) {
                    int dup = 0;
                    for (int j = 0; j < old_count; j++)
                        if (strcmp(g_catalog[j].full_name, tmp[i].full_name)==0) { dup=1; break; }
                    if (!dup) g_catalog[g_catalog_count++] = tmp[i];
                }
                if (g_catalog_count > 0) catalog_save_cache(g_catalog, g_catalog_count);
                /* Jump to last page to show new models */
                cur_page = 999; /* catalog_browse will clamp to last page */
                continue;
            } else if (action == 'U') {
                custom_url_flow();
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                continue;
            } else {
                break;
            }
        }
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
    int first_show = 1;
    while (1) {
        cli_clear();
        show_menu(&specs, first_show);
        first_show = 0;
        int choice = cli_prompt_choice(6);

        switch (choice) {
            case 1:
                cli_clear();
                chat_quick_start();
                break;
            case 2:
                cli_clear();
                specs_run_interactive();
                specs_load(&specs);
                printf("  Press Enter to continue...");
                fflush(stdout);
                { char _b[4]; fgets(_b, sizeof(_b), stdin); }
                break;
            case 3:
                cli_clear();
                browse_flow(&specs);
                break;
            case 4:
                cli_clear();
                my_models_flow();
                break;
            case 5: {
                cli_clear();
                chat_config_t cfg = chat_default_config();
                chat_run(&cfg);
                break;
            }
            case 6:
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
