/*
 * fllm CLI - Hardware detection & rating
 * Real hardware only. Enumerates all GPUs individually.
 */
#include "cli.h"
#include "cpu_features.h"

#ifdef _WIN32
#include <intrin.h>
#endif

/* ── RAM detection ────────────────────────────────────────────────── */

static double detect_ram_gb(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return (double)ms.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    return 0;
#elif defined(__APPLE__)
    FILE* fp = popen("sysctl -n hw.memsize 2>/dev/null", "r");
    if (fp) {
        unsigned long long bytes = 0;
        if (fscanf(fp, "%llu", &bytes) == 1) { pclose(fp); return bytes / (1024.0*1024.0*1024.0); }
        pclose(fp);
    }
    return 0;
#else
    FILE* fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            unsigned long kb;
            if (sscanf(line, "MemTotal: %lu kB", &kb) == 1) {
                fclose(fp);
                return kb / (1024.0 * 1024.0);
            }
        }
        fclose(fp);
    }
    return 0;
#endif
}

/* ── CPU name ─────────────────────────────────────────────────────── */

static void detect_cpu_name(char* buf, int sz) {
    buf[0] = '\0';
#ifdef _WIN32
    int info[4];
    char brand[49] = {0};
    __cpuid(info, 0x80000000);
    if ((unsigned)info[0] >= 0x80000004) {
        __cpuid((int*)(brand +  0), 0x80000002);
        __cpuid((int*)(brand + 16), 0x80000003);
        __cpuid((int*)(brand + 32), 0x80000004);
        char* p = brand;
        while (*p == ' ') p++;
        strncpy(buf, p, sz - 1);
    }
#elif defined(__APPLE__)
    FILE* fp = popen("sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
    if (fp) { fgets(buf, sz, fp); pclose(fp); buf[strcspn(buf, "\n")] = '\0'; }
#else
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* p = strchr(line, ':');
                if (p) { p++; while (*p == ' ') p++; strncpy(buf, p, sz - 1); buf[strcspn(buf, "\n")] = '\0'; }
                break;
            }
        }
        fclose(fp);
    }
#endif
    if (buf[0] == '\0') strncpy(buf, "Unknown CPU", sz - 1);
}

/* ── Helper: trim string in place ─────────────────────────────────── */

static void trim(char* s) {
    while (s[0] == ' ') memmove(s, s + 1, strlen(s));
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\r' || s[len-1] == '\n'))
        s[--len] = '\0';
}

/* ── Helper: guess if a GPU name is integrated ────────────────────── */

static int is_integrated_gpu(const char* name) {
    if (strstr(name, "Radeon(TM) Graphics")) return 1;
    if (strstr(name, "Radeon Graphics"))     return 1;
    if (strstr(name, "Radeon Vega"))         return 1;
    if (strstr(name, "Intel HD"))            return 1;
    if (strstr(name, "Intel UHD"))           return 1;
    if (strstr(name, "Intel Iris"))          return 1;
    if (strstr(name, "Intel(R) HD"))         return 1;
    if (strstr(name, "Intel(R) UHD"))        return 1;
    if (strstr(name, "Intel(R) Iris"))       return 1;
    if (strstr(name, "Microsoft Basic"))     return 1;
    return 0;
}

/* ── GPU detection ────────────────────────────────────────────────── */

static void detect_gpus(hw_specs_t* s) {
    s->gpu_count = 0;

#ifdef __APPLE__
    /* macOS: system_profiler lists all GPUs */
    FILE* fp = popen("system_profiler SPDisplaysDataType 2>/dev/null", "r");
    if (fp) {
        char line[256];
        gpu_info_t* g = NULL;
        while (fgets(line, sizeof(line), fp) && s->gpu_count < MAX_GPUS) {
            if (strstr(line, "Chipset Model:") || strstr(line, "Chip:")) {
                g = &s->gpus[s->gpu_count++];
                memset(g, 0, sizeof(*g));
                g->has_metal = 1;
                char* p = strchr(line, ':'); if (p) { p++; while (*p == ' ') p++; strncpy(g->name, p, sizeof(g->name)-1); trim(g->name); }
                g->is_discrete = !is_integrated_gpu(g->name);
                strncpy(g->driver, "Metal", sizeof(g->driver)-1);
            }
            if (g && strstr(line, "VRAM")) {
                char* p = strchr(line, ':');
                if (p) { p++; g->vram_gb = atof(p) / 1024.0; } /* MB to GB */
            }
        }
        pclose(fp);
    }
    /* Apple Silicon: unified memory GPU */
    if (s->gpu_count == 0) {
        gpu_info_t* g = &s->gpus[s->gpu_count++];
        memset(g, 0, sizeof(*g));
        snprintf(g->name, sizeof(g->name), "Apple GPU (unified)");
        g->vram_gb = s->ram_gb;
        g->has_metal = 1;
        g->is_discrete = 0;
        strncpy(g->driver, "Metal", sizeof(g->driver)-1);
    }

#elif defined(_WIN32)
    /* Windows: WMIC enumerates ALL GPUs (integrated + discrete) */
    FILE* fp = _popen("wmic path win32_VideoController get Name,AdapterRAM,DriverVersion /format:csv 2>nul", "r");
    if (fp) {
        char line[512];
        int header_done = 0;
        while (fgets(line, sizeof(line), fp) && s->gpu_count < MAX_GPUS) {
            if (strlen(line) < 5) continue;
            if (strstr(line, "AdapterRAM") || strstr(line, "DriverVersion")) { header_done = 1; continue; }
            if (!header_done) continue;

            /* CSV: Node,AdapterRAM,DriverVersion,Name */
            char* p = strchr(line, ',');
            if (!p) continue;
            p++; /* skip node */

            char* c1 = strchr(p, ','); if (!c1) continue; *c1 = '\0';
            double vram_bytes = atof(p);
            p = c1 + 1;

            char* c2 = strchr(p, ','); if (!c2) continue; *c2 = '\0';
            char driver[64] = {0};
            strncpy(driver, p, sizeof(driver)-1); trim(driver);
            p = c2 + 1;

            char name[128] = {0};
            strncpy(name, p, sizeof(name)-1); trim(name);

            if (strlen(name) < 3) continue;
            if (strstr(name, "Microsoft Basic")) continue;

            gpu_info_t* g = &s->gpus[s->gpu_count++];
            memset(g, 0, sizeof(*g));
            strncpy(g->name, name, sizeof(g->name)-1);
            g->vram_gb = vram_bytes / (1024.0 * 1024.0 * 1024.0);
            strncpy(g->driver, driver, sizeof(g->driver)-1);
            g->is_discrete = !is_integrated_gpu(name);
        }
        _pclose(fp);
    }

    /* nvidia-smi: get accurate VRAM + driver for NVIDIA GPUs, mark CUDA */
    fp = _popen("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>nul", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strlen(line) < 3) continue;
            char nv_name[128] = {0};
            double nv_vram = 0;
            char nv_driver[64] = {0};

            char* c1 = strchr(line, ',');
            if (c1) {
                *c1 = '\0';
                strncpy(nv_name, line, sizeof(nv_name)-1); trim(nv_name);
                char* c2 = strchr(c1+1, ',');
                if (c2) {
                    *c2 = '\0';
                    nv_vram = atof(c1+1) / 1024.0; /* MiB to GB */
                    strncpy(nv_driver, c2+1, sizeof(nv_driver)-1); trim(nv_driver);
                } else {
                    nv_vram = atof(c1+1) / 1024.0;
                }
            }

            /* Find matching GPU in our list and update it */
            int found = 0;
            for (int i = 0; i < s->gpu_count; i++) {
                if (strstr(s->gpus[i].name, "NVIDIA") || strstr(s->gpus[i].name, "GeForce") || strstr(s->gpus[i].name, "RTX") || strstr(s->gpus[i].name, "GTX")) {
                    s->gpus[i].vram_gb = nv_vram;
                    s->gpus[i].has_cuda = 1;
                    s->gpus[i].is_discrete = 1;
                    if (nv_driver[0]) strncpy(s->gpus[i].driver, nv_driver, sizeof(s->gpus[i].driver)-1);
                    found = 1;
                    break;
                }
            }
            if (!found && s->gpu_count < MAX_GPUS) {
                gpu_info_t* g = &s->gpus[s->gpu_count++];
                memset(g, 0, sizeof(*g));
                strncpy(g->name, nv_name, sizeof(g->name)-1);
                g->vram_gb = nv_vram;
                strncpy(g->driver, nv_driver, sizeof(g->driver)-1);
                g->has_cuda = 1;
                g->is_discrete = 1;
            }
        }
        _pclose(fp);
    }

#else
    /* Linux: enumerate all GPUs via lspci, then enrich with nvidia-smi / rocm-smi */

    FILE* fp = popen("lspci 2>/dev/null | grep -iE 'vga|3d|display'", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp) && s->gpu_count < MAX_GPUS) {
            char* colon = strstr(line, ": ");
            if (!colon) continue;
            colon += 2;

            gpu_info_t* g = &s->gpus[s->gpu_count++];
            memset(g, 0, sizeof(*g));
            strncpy(g->name, colon, sizeof(g->name)-1); trim(g->name);
            g->is_discrete = !is_integrated_gpu(g->name);
        }
        pclose(fp);
    }

    /* nvidia-smi for NVIDIA GPUs */
    fp = popen("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        char line[256];
        int nv_idx = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (strlen(line) < 3) continue;
            char nv_name[128]={0}; double nv_vram=0; char nv_drv[64]={0};
            char* c1 = strchr(line, ',');
            if (c1) {
                *c1='\0'; strncpy(nv_name, line, 127); trim(nv_name);
                char* c2 = strchr(c1+1, ',');
                if (c2) { *c2='\0'; nv_vram=atof(c1+1)/1024.0; strncpy(nv_drv,c2+1,63); trim(nv_drv); }
                else nv_vram=atof(c1+1)/1024.0;
            }
            /* Match to lspci entry */
            for (int i = 0; i < s->gpu_count; i++) {
                if ((strstr(s->gpus[i].name, "NVIDIA") || strstr(s->gpus[i].name, "GeForce")) && !s->gpus[i].has_cuda) {
                    s->gpus[i].vram_gb = nv_vram;
                    s->gpus[i].has_cuda = 1;
                    s->gpus[i].is_discrete = 1;
                    if (nv_drv[0]) strncpy(s->gpus[i].driver, nv_drv, sizeof(s->gpus[i].driver)-1);
                    break;
                }
            }
            nv_idx++;
        }
        pclose(fp);
    }

    /* rocm-smi for AMD discrete GPUs */
    fp = popen("rocm-smi --showproductname 2>/dev/null | grep -i 'card'", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            for (int i = 0; i < s->gpu_count; i++) {
                if (strstr(s->gpus[i].name, "AMD") && s->gpus[i].is_discrete && !s->gpus[i].has_rocm) {
                    s->gpus[i].has_rocm = 1;
                    break;
                }
            }
        }
        pclose(fp);
    }
#endif

    if (s->gpu_count == 0) {
        gpu_info_t* g = &s->gpus[0];
        memset(g, 0, sizeof(*g));
        strncpy(g->name, "None detected", sizeof(g->name)-1);
        s->gpu_count = 1;
    }
}

/* ── Detect all hardware ──────────────────────────────────────────── */

void specs_detect(hw_specs_t* s) {
    memset(s, 0, sizeof(*s));
    s->valid = 1;

    cpu_features_t cpu = detect_cpu_features();
    s->cpu_cores   = cpu.num_cores;
    s->cpu_threads = cpu.num_threads;
    s->has_avx2    = cpu.has_avx2;
    s->has_avx512  = cpu.has_avx512f;
    s->ram_gb      = detect_ram_gb();
    detect_cpu_name(s->cpu_name, sizeof(s->cpu_name));
    detect_gpus(s);
}

/* ── Best discrete GPU VRAM ───────────────────────────────────────── */

static double best_discrete_vram(const hw_specs_t* s) {
    double best = 0;
    for (int i = 0; i < s->gpu_count; i++)
        if (s->gpus[i].is_discrete && s->gpus[i].vram_gb > best)
            best = s->gpus[i].vram_gb;
    return best;
}

static int has_any_compute(const hw_specs_t* s) {
    for (int i = 0; i < s->gpu_count; i++)
        if (s->gpus[i].has_cuda || s->gpus[i].has_rocm || s->gpus[i].has_metal)
            return 1;
    return 0;
}

/* ── Rate hardware ────────────────────────────────────────────────── */

void specs_rate(hw_specs_t* s) {
    double dgpu_vram = best_discrete_vram(s);

    /* Tier: based on RAM + cores. GPU VRAM doesn't inflate tier for CPU inference. */
    if (s->ram_gb >= 32 && s->cpu_cores >= 12)      s->tier = TIER_ULTRA;
    else if (s->ram_gb >= 16 && s->cpu_cores >= 8)   s->tier = TIER_HIGH;
    else if (s->ram_gb >= 8  && s->cpu_cores >= 4)   s->tier = TIER_MEDIUM;
    else                                              s->tier = TIER_LOW;

    /* Stars */
    int score = 0;

    if (s->ram_gb >= 64) score += 4;
    else if (s->ram_gb >= 32) score += 3;
    else if (s->ram_gb >= 16) score += 2;
    else if (s->ram_gb >= 8) score += 1;

    if (s->cpu_cores >= 16) score += 3;
    else if (s->cpu_cores >= 8) score += 2;
    else if (s->cpu_cores >= 4) score += 1;

    if (s->has_avx2) score += 1;
    if (s->has_avx512) score += 1;

    /* GPU bonus: only big discrete GPUs matter */
    if (dgpu_vram >= 16.0) score += 3;
    else if (dgpu_vram >= 8.0) score += 2;
    else if (dgpu_vram >= 6.0) score += 1;
    /* 4 GB 1650 = no bonus, that's entry-level */

    if (score >= 11)     s->stars = 5;
    else if (score >= 8) s->stars = 4;
    else if (score >= 6) s->stars = 3;
    else if (score >= 3) s->stars = 2;
    else                 s->stars = 1;
}

/* ── Save / Load ──────────────────────────────────────────────────── */

void specs_save(const hw_specs_t* s) {
    char path[512];
    cli_get_specs_path(path, sizeof(path));
    cli_ensure_dirs();
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(s, sizeof(*s), 1, f); fclose(f); }
}

int specs_load(hw_specs_t* s) {
    char path[512];
    cli_get_specs_path(path, sizeof(path));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int ok = (fread(s, sizeof(*s), 1, f) == 1) && s->valid;
    fclose(f);
    return ok;
}

/* ── Print ────────────────────────────────────────────────────────── */

void specs_print(const hw_specs_t* s) {
    cli_separator();
    cli_cprint("  HARDWARE REPORT\n", CLR_CYAN);
    cli_separator();
    cli_delay(200);

    printf("\n");
    cli_cprint("  CPU\n", CLR_WHITE);
    cli_delay(150);
    printf("  Name:      %s\n", s->cpu_name); fflush(stdout); cli_delay(100);
    printf("  Cores:     %d (%d threads)\n", s->cpu_cores, s->cpu_threads); fflush(stdout); cli_delay(100);
    printf("  AVX2:      %s\n", s->has_avx2 ? "Yes" : "No"); fflush(stdout); cli_delay(100);
    printf("  AVX-512:   %s\n", s->has_avx512 ? "Yes" : "No"); fflush(stdout); cli_delay(100);

    printf("\n");
    cli_cprint("  MEMORY\n", CLR_WHITE);
    cli_delay(150);
    printf("  RAM:       %.1f GB\n", s->ram_gb); fflush(stdout); cli_delay(100);

    printf("\n");
    cli_cprint("  GPU(s)\n", CLR_WHITE);
    cli_delay(150);

    for (int i = 0; i < s->gpu_count; i++) {
        const gpu_info_t* g = &s->gpus[i];

        if (s->gpu_count > 1) {
            printf("  ");
            if (g->is_discrete)
                cli_cprint("[Discrete]  ", CLR_GREEN);
            else
                cli_cprint("[Integrated]", CLR_YELLOW);
            printf(" ");
        }

        printf("%s\n", g->name); fflush(stdout); cli_delay(80);

        if (g->vram_gb > 0.1) {
            printf("             VRAM: %.1f GB\n", g->vram_gb); fflush(stdout); cli_delay(80);
        }
        if (g->driver[0]) {
            printf("             Driver: %s\n", g->driver); fflush(stdout); cli_delay(80);
        }

        if (g->has_cuda || g->has_rocm || g->has_metal) {
            printf("             Compute: ");
            int first = 1;
            if (g->has_cuda)  { cli_cprint("CUDA", CLR_GREEN); first = 0; }
            if (g->has_rocm)  { if (!first) printf(", "); cli_cprint("ROCm", CLR_GREEN); first = 0; }
            if (g->has_metal) { if (!first) printf(", "); cli_cprint("Metal", CLR_GREEN); }
            printf("\n"); fflush(stdout); cli_delay(80);
        }

        if (i < s->gpu_count - 1) printf("\n");
    }

    if (!has_any_compute(s)) {
        printf("  Compute:   ");
        cli_cprint("CPU only (no GPU acceleration detected)\n", CLR_YELLOW);
        fflush(stdout); cli_delay(80);
    }

    printf("\n");
    cli_delay(300);
    cli_separator();
    printf("  Rating:    ");
    cli_stars(s->stars);
    printf("  %s tier\n", cli_tier_name(s->tier));
    cli_separator();
    fflush(stdout);
    cli_delay(200);

    printf("\n");
    switch (s->tier) {
        case TIER_LOW:
            cli_cprint("  Can run: small models (up to ~1B params)\n", CLR_YELLOW);
            break;
        case TIER_MEDIUM:
            cli_cprint("  Can run: medium models (up to ~3B params)\n", CLR_GREEN);
            break;
        case TIER_HIGH:
            cli_cprint("  Can run: large models (up to ~8B params)\n", CLR_GREEN);
            break;
        case TIER_ULTRA:
            cli_cprint("  Can run: any model comfortably (8B+ no problem)\n", CLR_GREEN);
            break;
    }
    printf("\n");
    fflush(stdout);
}

/* ── Interactive specs check ──────────────────────────────────────── */

void specs_run_interactive(void) {
    hw_specs_t s;

    printf("\n");
    cli_cprint("  Scanning hardware", CLR_CYAN);
    fflush(stdout);

    /* Animated dots while detecting */
    specs_detect(&s);

    /* Show progress dots with delays to feel like it's working */
    for (int i = 0; i < 3; i++) {
        cli_delay(300);
        printf(".");
        fflush(stdout);
    }
    cli_delay(200);
    printf(" done.\n\n");
    fflush(stdout);
    cli_delay(400);

    specs_rate(&s);
    specs_save(&s);
    specs_print(&s);
    cli_cprint("  Specs saved. Model recommendations will use these.\n\n", CLR_GREEN);
    fflush(stdout);
}
