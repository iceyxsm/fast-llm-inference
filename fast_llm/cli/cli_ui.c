/*
 * fllm CLI - UI helpers, path utils, timing
 */
#include "cli.h"

/* ── Path helpers ─────────────────────────────────────────────────── */

void cli_get_fllm_dir(char* buf, int sz) {
    const char* home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (!home) home = ".";
    snprintf(buf, sz, "%s%c%s", home, PATH_SEP, FLLM_DIR_NAME);
}

void cli_get_models_dir(char* buf, int sz) {
    char dir[1024];
    cli_get_fllm_dir(dir, sizeof(dir));
    snprintf(buf, sz, "%s%c%s", dir, PATH_SEP, MODELS_DIR_NAME);
}

void cli_get_pid_path(char* buf, int sz) {
    char dir[1024];
    cli_get_fllm_dir(dir, sizeof(dir));
    snprintf(buf, sz, "%s%c%s", dir, PATH_SEP, PID_FILE_NAME);
}

void cli_get_specs_path(char* buf, int sz) {
    char dir[1024];
    cli_get_fllm_dir(dir, sizeof(dir));
    snprintf(buf, sz, "%s%c%s", dir, PATH_SEP, SPECS_FILE_NAME);
}

void cli_get_log_path(char* buf, int sz) {
    char dir[1024];
    cli_get_fllm_dir(dir, sizeof(dir));
    snprintf(buf, sz, "%s%c%s", dir, PATH_SEP, LOG_FILE_NAME);
}

void cli_get_catalog_path(char* buf, int sz) {
    char dir[1024];
    cli_get_fllm_dir(dir, sizeof(dir));
    snprintf(buf, sz, "%s%c%s", dir, PATH_SEP, CATALOG_CACHE);
}

void cli_ensure_dirs(void) {
    char d[1024];
    cli_get_fllm_dir(d, sizeof(d));   MKDIR_P(d);
    cli_get_models_dir(d, sizeof(d));  MKDIR_P(d);
}

int cli_file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* ── Color output ─────────────────────────────────────────────────── */

void cli_color(int color) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(h, (WORD)color);
#else
    const char* a;
    switch (color) {
        case CLR_GREEN:  a = "\033[32m"; break;
        case CLR_CYAN:   a = "\033[36m"; break;
        case CLR_RED:    a = "\033[31m"; break;
        case CLR_YELLOW: a = "\033[33m"; break;
        case CLR_WHITE:  a = "\033[97m"; break;
        default:         a = "\033[0m";  break;
    }
    fputs(a, stdout);
#endif
}

void cli_reset(void) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(h, 7);
#else
    fputs("\033[0m", stdout);
#endif
}

void cli_cprint(const char* text, int color) {
    cli_color(color);
    fputs(text, stdout);
    cli_reset();
}

/* ── UI widgets ───────────────────────────────────────────────────── */

void cli_banner(void) {
    printf("\n");
    cli_cprint("  ╔══════════════════════════════════╗\n", CLR_CYAN);
    cli_cprint("  ║        fllm - Fast LLM CLI       ║\n", CLR_CYAN);
    cli_cprint("  ╚══════════════════════════════════╝\n", CLR_CYAN);
    printf("\n");
}

void cli_separator(void) {
    cli_cprint("  ──────────────────────────────────\n", CLR_CYAN);
}

void cli_clear(void) {
    /* Full screen clear, then redraw banner at top */
#ifdef _WIN32
    system("cls");
#else
    printf("\033[2J\033[H");
    fflush(stdout);
#endif
    cli_banner();
}

void cli_stars(int n) {
    if (n < 1) n = 1;
    if (n > 5) n = 5;
    int color = (n >= 4) ? CLR_GREEN : (n >= 3) ? CLR_YELLOW : CLR_RED;
    cli_color(color);
    for (int i = 0; i < n; i++) printf("★");
    for (int i = n; i < 5; i++) printf("☆");
    cli_reset();
}

const char* cli_tier_name(hw_tier_t t) {
    switch (t) {
        case TIER_LOW:    return "Low";
        case TIER_MEDIUM: return "Medium";
        case TIER_HIGH:   return "High";
        case TIER_ULTRA:  return "Ultra";
    }
    return "Unknown";
}

int cli_prompt_yn(const char* msg) {
    printf("  %s [Y/n]: ", msg);
    fflush(stdout);
    char buf[16] = {0};
    if (!fgets(buf, sizeof(buf), stdin)) return 0;
    return (buf[0] != 'n' && buf[0] != 'N');
}

int cli_prompt_choice(int max) {
    printf("\n  > ");
    fflush(stdout);
    char buf[16] = {0};
    if (!fgets(buf, sizeof(buf), stdin)) return -1;
    int c = atoi(buf);
    if (c < 0 || c > max) return -1;
    return c;
}

/* ── Timing ───────────────────────────────────────────────────────── */

double cli_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

/* ── Delay ────────────────────────────────────────────────────────── */

void cli_delay(int ms) {
    SLEEP_MS(ms);
}

/* ── Slow print: prints line by line with a delay ─────────────────── */

void cli_print_slow(const char* text, int delay_ms) {
    /* Print char by char for short strings, line by line for long */
    const char* p = text;
    while (*p) {
        putchar(*p);
        fflush(stdout);
        if (*p == '\n') SLEEP_MS(delay_ms);
        p++;
    }
}
