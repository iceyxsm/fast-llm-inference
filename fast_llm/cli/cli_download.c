/*
 * fllm CLI - Download manager
 */
#include "cli.h"

/* ── Download a file via curl/wget ────────────────────────────────── */

int download_file(const char* url, const char* dest) {
    char cmd[2048];

#ifdef _WIN32
    snprintf(cmd, sizeof(cmd),
        "curl.exe -L --progress-bar -o \"%s\" \"%s\"", dest, url);
#else
    snprintf(cmd, sizeof(cmd),
        "curl -L --progress-bar -o '%s' '%s' 2>&1 || "
        "wget --show-progress -O '%s' '%s' 2>&1",
        dest, url, dest, url);
#endif

    int ret = system(cmd);
    return (ret == 0 && cli_file_exists(dest));
}

/* ── Download a catalog model ─────────────────────────────────────── */

int download_model(const model_entry_t* m) {
    char mdir[512], dest[512], size_str[32];
    cli_get_models_dir(mdir, sizeof(mdir));
    cli_ensure_dirs();
    snprintf(dest, sizeof(dest), "%s%c%s", mdir, PATH_SEP, m->filename);

    if (m->size_mb >= 1024)
        snprintf(size_str, sizeof(size_str), "%.1f GB", m->size_mb / 1024.0);
    else if (m->size_mb > 0)
        snprintf(size_str, sizeof(size_str), "%.0f MB", m->size_mb);
    else
        snprintf(size_str, sizeof(size_str), "unknown size");

    printf("\n");
    cli_cprint("  Downloading: ", CLR_CYAN);
    printf("%s\n", m->short_name);
    printf("  Size: %s\n", size_str);
    printf("  To:   %s\n\n", dest);

    if (!download_file(m->url, dest)) {
        cli_cprint("  Download failed!\n", CLR_RED);
        printf("  Make sure curl or wget is installed.\n");
        printf("  Manual: %s\n\n", m->url);
        return 0;
    }

    printf("\n  ");
    cli_cprint("Download complete!\n\n", CLR_GREEN);
    return 1;
}

/* ── Find any local .gguf model ───────────────────────────────────── */

int find_local_model(char* out, int sz) {
    char mdir[512];
    cli_get_models_dir(mdir, sizeof(mdir));

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char pat[512];
    snprintf(pat, sizeof(pat), "%s\\*.gguf", mdir);
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        snprintf(out, sz, "%s\\%s", mdir, fd.cFileName);
        FindClose(h);
        return 1;
    }
#else
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "ls '%s'/*.gguf 2>/dev/null | head -1", mdir);
    FILE* fp = popen(cmd, "r");
    if (fp) {
        if (fgets(out, sz, fp)) {
            out[strcspn(out, "\n")] = '\0';
            pclose(fp);
            if (strlen(out) > 0) return 1;
        } else { pclose(fp); }
    }
#endif
    out[0] = '\0';
    return 0;
}

/* ── List all local .gguf models ──────────────────────────────────── */

int list_local_models(char out[][512], int max) {
    char mdir[512];
    cli_get_models_dir(mdir, sizeof(mdir));
    int count = 0;

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char pat[512];
    snprintf(pat, sizeof(pat), "%s\\*.gguf", mdir);
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (count >= max) break;
            snprintf(out[count], 512, "%s\\%s", mdir, fd.cFileName);
            count++;
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "ls '%s'/*.gguf 2>/dev/null", mdir);
    FILE* fp = popen(cmd, "r");
    if (fp) {
        char line[512];
        while (fgets(line, sizeof(line), fp) && count < max) {
            line[strcspn(line, "\n")] = '\0';
            if (strlen(line) > 0) {
                strncpy(out[count], line, 511);
                count++;
            }
        }
        pclose(fp);
    }
#endif
    return count;
}
